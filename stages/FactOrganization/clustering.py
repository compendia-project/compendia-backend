import math
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.express as px
import umap
from dotenv import load_dotenv
from kneed import KneeLocator
from nltk.corpus import stopwords
from scipy import spatial
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from common.config import CLUSTER_CONFIGS
from common.gpt_helper import GPTHelper
from common.utils import console
from common.utils.timing_logger import LOGGER, log_execution_time

load_dotenv()

gpt_helper = GPTHelper()

# Initialize stopwords with error handling
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    import nltk

    try:
        nltk.download("stopwords")
        STOPWORDS = set(stopwords.words("english"))
    except:
        # Fallback to a basic set of common English stopwords
        STOPWORDS = set(
            [
                "i",
                "me",
                "my",
                "myself",
                "we",
                "our",
                "ours",
                "ourselves",
                "you",
                "your",
                "yours",
                "yourself",
                "yourselves",
                "he",
                "him",
                "his",
                "himself",
                "she",
                "her",
                "hers",
                "herself",
                "it",
                "its",
                "itself",
                "they",
                "them",
                "their",
                "theirs",
                "themselves",
                "what",
                "which",
                "who",
                "whom",
                "this",
                "that",
                "these",
                "those",
                "am",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "having",
                "do",
                "does",
                "did",
                "doing",
                "a",
                "an",
                "the",
                "and",
                "but",
                "if",
                "or",
                "because",
                "as",
                "until",
                "while",
                "of",
                "at",
                "by",
                "for",
                "with",
                "through",
                "during",
                "before",
                "after",
                "above",
                "below",
                "up",
                "down",
                "in",
                "out",
                "on",
                "off",
                "over",
                "under",
                "again",
                "further",
                "then",
                "once",
            ]
        )


def process_text(text, query_embedding):
    try:
        embedding = gpt_helper.get_embeddings(text)
        relatedness = calc_relatedness(query_embedding, embedding)
        print("relatedness", relatedness)
        return embedding, relatedness
    except Exception as e:
        console.print(f"Error getting embedding for text: {e}")
        return np.zeros(1536), 0.0


def get_all_embeddings(texts, query):
    query_embedding = gpt_helper.get_embeddings(query)
    embeddings = []
    similarities = []

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=min(32, len(texts))) as executor:
        # Submit all tasks
        future_to_text = {
            executor.submit(process_text, text, query_embedding): text for text in texts
        }

        # Process results as they complete
        for future in as_completed(future_to_text):
            embedding, similarity = future.result()
            embeddings.append(embedding)
            similarities.append(similarity)

    return np.array(embeddings), similarities


def temporySimilarityCalculation(clusterData, query):
    all_mean_similarities = []
    cluster_summaries = clusterData["cluster_summary"]
    for cluster in clusterData["cluster_wise_facts"]:
        cluster_id = cluster["cluster_id"]
        texts = []
        for fact in cluster["facts"]:
            texts.append(fact["fact_content"])
        embeddings, similarities = get_all_embeddings(texts, query)
        mean_similarity = np.mean(similarities)
        all_mean_similarities.append(mean_similarity)
        cluster_summaries[str(cluster_id)]["mean_similarity"] = mean_similarity

    if all_mean_similarities:
        min_sim = min(all_mean_similarities)
        max_sim = max(all_mean_similarities)
        range_sim = max_sim - min_sim

        # Update the scaled values in cluster_summaries
        for cluster_id in cluster_summaries:
            original_sim = cluster_summaries[cluster_id]["mean_similarity"]
            if range_sim > 0:
                scaled_sim = (original_sim - min_sim) / range_sim
            else:
                scaled_sim = 0.0  # If all values are the same
            cluster_summaries[cluster_id]["mean_similarity_normalized"] = float(
                scaled_sim
            )
    return clusterData


def reduce_dimensions(X, method="pca"):
    """Use PCA, t-SNE or UMAP for dimensionality reduction."""
    if method == "pca":
        pca = PCA(n_components=min(len(X), 10))
        X = pca.fit_transform(X)
    elif method == "tsne":
        tsne = TSNE(n_components=2, random_state=42)
        X = tsne.fit_transform(X)
    elif method == "umap":
        n_neighbors = min(15, X.shape[0] - 1) if X.shape[0] > 1 else 2
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        X = reducer.fit_transform(X)
        # reducer = umap.UMAP(n_components=2, random_state=42)
        # X = reducer.fit_transform(X)
    return X


def plot_umap(embeddings_2d, labels, facts, file_path):
    filename = f"{file_path}/umap_visualization.html"
    # Prepare hover text
    hover_texts = [
        f"Cluster: {label}<br>{fact['fact_content'][:150]}..."
        for label, fact in zip(labels, facts)
    ]

    # Create DataFrame for plotting
    plot_df = pd.DataFrame(
        {
            "x": embeddings_2d[:, 0],
            "y": embeddings_2d[:, 1],
            "cluster": labels.astype(str),
            "hover_text": hover_texts,
        }
    )

    # Create interactive plot
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="cluster",
        hover_name="hover_text",
        title="Fact Clusters UMAP Visualization",
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        labels={"color": "Cluster ID"},
    )

    # Customize layout
    fig.update_layout(
        hoverlabel=dict(bgcolor="white", font_size=12),
        plot_bgcolor="rgba(240,240,240,0.9)",
        width=1200,
        height=800,
        title_x=0.5,
    )

    # Save and show
    fig.write_html(filename)
    console.print(f"[bold green]UMAP visualization saved to {filename}[/bold green]")
    return fig


def get_k_value(num_facts):
    print("Num Facts: ", num_facts)
    max_clusters = CLUSTER_CONFIGS["MAX_CLUSTER_SIZE"]
    assumed_clusters = math.ceil(num_facts / 6) + 1
    print("Assumed Clusters: ", assumed_clusters)
    if max_clusters < assumed_clusters:
        print("Max Clusters: ", max_clusters)
        return max_clusters
    print("Assumed Clusters: ", assumed_clusters)
    return assumed_clusters


@log_execution_time
def cluster_facts(facts, file_path, query, threshold=0.2):
    if not facts:
        return {"error": "No facts provided for clustering."}
    texts = [d["fact_content"] for d in facts]

    console.print("[bold yellow]Getting Embeddings...[/bold yellow]")
    X, similarities = get_all_embeddings(texts, query)
    if len(X) == 0 or X.shape[0] == 0:
        return {"error": "Failed to generate embeddings for all facts."}
    console.print("[bold green]Getting Embeddings Completed![/bold green]")

    # Normalize & Reduce Dimensions using RobustScaler and UMAP
    # X = RobustScaler().fit_transform(X)  # Use RobustScaler instead of StandardScaler
    # X = reduce_dimensions(X, method="umap")  # Change method to 'pca', 'tsne', or 'umap'

    try:
        X = RobustScaler().fit_transform(X)
        X = reduce_dimensions(X, method="umap")
    except ValueError as e:
        return {"error": f"Dimensionality reduction failed: {str(e)}"}

    console.print("[bold green]Getting Embeddings Completed![/bold green]")

    # Finding Optimal k
    console.print("[bold yellow]K value[/bold yellow] ", get_k_value(len(X)))

    k_values = range(2, get_k_value(len(X)))
    neg_log_likelihoods = []
    models = {}

    def process_k(k):
        console.print(f"k = {k}")
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
        gmm.fit(X)
        total_log_likelihood = gmm.score(X) * X.shape[0]
        return k, -total_log_likelihood, gmm

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=min(32, len(k_values))) as executor:
        # Submit all tasks
        future_to_k = {executor.submit(process_k, k): k for k in k_values}

        # Process results as they complete
        for future in as_completed(future_to_k):
            k, neg_log_likelihood, gmm = future.result()
            neg_log_likelihoods.append(neg_log_likelihood)
            models[k] = gmm

    kneedle = KneeLocator(
        k_values, neg_log_likelihoods, curve="convex", direction="increasing"
    )
    best_k = kneedle.elbow or min(k_values, key=lambda k: models[k].bic(X))

    console.print(
        f"[bold green]Best number of clusters (k) selected: {best_k}[/bold green]"
    )

    final_gmm = models[best_k]
    probabilities = final_gmm.predict_proba(X)
    labels = final_gmm.predict(X)

    # Fact-wise Assignments
    fact_wise_clusters = []
    cluster_dict = defaultdict(list)

    for i, fact in enumerate(facts):
        assigned_clusters = [
            {"cluster_id": cluster_id + 1, "probability": float(prob)}
            for cluster_id, prob in enumerate(probabilities[i])
            if prob >= threshold
        ]
        fact_wise_clusters.append(
            {
                "fact_id": fact["fact_id"],
                "fact_content": fact["fact_content"],
                "assigned_clusters": assigned_clusters,
            }
        )

        for cluster in assigned_clusters:
            cluster_dict[cluster["cluster_id"]].append(
                {"fact_id": fact["fact_id"], "fact_content": fact["fact_content"]}
            )

    cluster_wise_facts = [
        {"cluster_id": cluster_id, "cluster_size": len(facts), "facts": facts}
        for cluster_id, facts in cluster_dict.items()
    ]

    # Clustering Evaluation Metrics
    def process_evaluation(X, labels, probabilities):
        return evaluate_clustering(X, labels, probabilities)

    def process_summarization(X, labels, cluster_dict, similarities):
        return summarize_clusters(X, labels, cluster_dict, similarities)

    def process_visualization(X, labels, facts, file_path):
        return plot_umap(X, labels, facts, file_path)

    # Run operations in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        eval_future = executor.submit(process_evaluation, X, labels, probabilities)
        summary_future = executor.submit(
            process_summarization, X, labels, cluster_dict, similarities
        )
        viz_future = executor.submit(process_visualization, X, labels, facts, file_path)

        # Get results
        eval_metrics = eval_future.result()
        cluster_summary = summary_future.result()
        viz_future.result()  # We don't need to store the return value of plot_umap

    # Combine original clusters and sub-clusters into a final result
    return {
        "cluster_wise_facts": cluster_wise_facts,
        "fact_wise_clusters": fact_wise_clusters,
        "evaluation_metrics": eval_metrics,
        "cluster_summary": cluster_summary,
    }


def evaluate_clustering(X, labels, probabilities):
    silhouette = silhouette_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    cluster_entropy = entropy(probabilities, axis=1).mean()

    # Convert numpy.float32 to regular Python float
    silhouette = float(silhouette)
    db_score = float(db_score)
    cluster_entropy = float(cluster_entropy)

    console.print(f"[bold cyan]Silhouette Score:[/bold cyan] {silhouette:.4f}")
    console.print(f"[bold cyan]Davies-Bouldin Index:[/bold cyan] {db_score:.4f}")
    console.print(f"[bold cyan]Cluster Entropy:[/bold cyan] {cluster_entropy:.4f}")

    return {
        "silhouette_score": silhouette,
        "davies_bouldin_score": db_score,
        "cluster_entropy": cluster_entropy,
    }


def summarize_clusters(X, labels, cluster_dict, similarities):
    # Calculate cluster centers in parallel
    def calculate_cluster_center(cluster_id):
        cluster_points = X[labels == cluster_id]
        return int(cluster_id), cluster_points.mean(axis=0)

    cluster_centers = {}
    with ThreadPoolExecutor(max_workers=min(32, len(np.unique(labels)))) as executor:
        future_to_cluster = {
            executor.submit(calculate_cluster_center, int(cluster_id)): cluster_id
            for cluster_id in np.unique(labels)
        }
        for future in as_completed(future_to_cluster):
            cluster_id, center = future.result()
            cluster_centers[cluster_id] = center

    # Process cluster summaries in parallel
    def process_cluster(cluster_id, center):
        cluster_facts = [
            fact["fact_content"] for fact in cluster_dict.get(cluster_id + 1, [])
        ]
        distances = np.linalg.norm(X[labels == cluster_id] - center, axis=1)
        closest_fact_idx = np.argmin(distances)
        representative_fact = (
            cluster_facts[closest_fact_idx] if cluster_facts else "N/A"
        )
        common_words = extract_keywords(cluster_facts)

        # Calculate mean similarity for this cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_similarities = [similarities[i] for i in cluster_indices]
        mean_similarity = np.mean(cluster_similarities) if cluster_similarities else 0.0

        return int(cluster_id), {
            "representative_fact": representative_fact,
            "top_keywords": common_words,
            "mean_similarity": float(mean_similarity),
        }

    # First collect all mean similarities
    all_mean_similarities = []
    cluster_summaries = {}

    with ThreadPoolExecutor(max_workers=min(32, len(cluster_centers))) as executor:
        future_to_cluster = {
            executor.submit(process_cluster, int(cluster_id), center): cluster_id
            for cluster_id, center in cluster_centers.items()
        }
        for future in as_completed(future_to_cluster):
            cluster_id, summary = future.result()
            cluster_summaries[cluster_id + 1] = summary
            all_mean_similarities.append(summary["mean_similarity"])

    # Scale all mean similarities to 0-1 range
    if all_mean_similarities:
        min_sim = min(all_mean_similarities)
        max_sim = max(all_mean_similarities)
        range_sim = max_sim - min_sim

        # Update the scaled values in cluster_summaries
        for cluster_id in cluster_summaries:
            original_sim = cluster_summaries[cluster_id]["mean_similarity"]
            if range_sim > 0:
                scaled_sim = (original_sim - min_sim) / range_sim
            else:
                scaled_sim = 0.0  # If all values are the same
            cluster_summaries[cluster_id]["mean_similarity_normalized"] = float(
                scaled_sim
            )

    return cluster_summaries


def extract_keywords(facts, top_n=5):
    words = re.findall(r"\w+", " ".join(facts).lower()) if facts else []
    filtered_words = [word for word in words if word not in STOPWORDS]
    common_words = [word for word, _ in Counter(filtered_words).most_common(top_n)]
    return common_words


def calc_relatedness(query_embedding, item_embedding):
    relatedness_fn = lambda x, y: 1 - spatial.distance.cosine(x, y)
    return relatedness_fn(query_embedding, item_embedding)
