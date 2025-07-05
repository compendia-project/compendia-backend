import html
import json
import os
import re
import webbrowser
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from bs4 import BeautifulSoup
from common.config import DATE_CONFIGS, PROMPTS, TEMPLATE_CONFIGS, THEME_CONFIGS
from common.gpt_helper import GPTHelper
from common.utils import console, convert_date
from common.utils.timing_logger import LOGGER, log_execution_time
from models.models import Overview, StoryLine, StyledNarrative
from scipy import spatial

gpt_helper = GPTHelper()


@log_execution_time
def style_narrative(fact):
    system_prompt = PROMPTS["STYLE_NARRATIVE"]

    user_prompt = f"""
        #### 
        {fact}
        ####
    """

    console.print("[bold yellow]Styling Narrative...[/bold yellow]")
    styled_narrative = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=StyledNarrative
    )
    try:
        styled_narrative = json.loads(styled_narrative)
        console.print("[bold green]Styling Narrative Completed![/bold green]")
        return styled_narrative
    except Exception as e:
        LOGGER.error(f"Error: {e}")
        console.print(f"[bold red]Error: {e}[/bold red]")
        return None


@log_execution_time
def refine_narrative(fact):
    system_prompt = PROMPTS["REFINE_NARRATIVE"]

    user_prompt = f"""
        #### 
        {fact}
        ####
    """

    console.print("[bold yellow]Styling Narrative...[/bold yellow]")
    styled_narrative = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=StyledNarrative
    )
    try:
        styled_narrative = json.loads(styled_narrative)
        console.print("[bold green]Styling Narrative Completed![/bold green]")
        return styled_narrative
    except Exception as e:
        LOGGER.error(f"Error: {e}")
        console.print(f"[bold red]Error: {e}[/bold red]")
        return None


@log_execution_time
def format_overview(summary):
    system_prompt = """
        You are a creative journalist. 
        You are given a text summary between #### delimeters.
        Your task is to  format the given summary according to the gicen example format in html. 
        Strickly follow the exmple format in html and provide html output. Use colors from ["#264653", "#27515B", "#287271", "#298880", "#2A988C"].
        
        Example:
        "start_point": "<p class="text-gray-800" style="flex-direction: column; align-items: center; font-size: 20px; text-align: center;">
                    This is an <span class="font-bold" style="color: #ff9999; font-size: 25px;">Example</span> of a stunning  <span class="font-bold" style="color: #fd7f6f; font-size: 25px;">starting point</span>.
                </p>"
    """

    user_prompt = f"""
        #### 
        {summary}
        ####

    """

    console.print("[bold yellow]Formatting Overview...[/bold yellow]")
    vis_refined = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=Overview
    )
    console.print("[bold green]Formating Overview Completed![/bold green]")
    return json.loads(vis_refined)


@log_execution_time
def create_storyline(paragraphs, search_query):
    system_prompt = """
        You are a creative journalist. 
        You are given set of data facts between #### delimeters and user search query between **** delimiters.
        In the data, you have the starting point of the story. 
        Your task is to organise the data facts into a cohesive and stunning storyline to the data story which expands the start_point.
        Do not change the format of the start_point. 
        
        Let's think step by step: 
        - Read the data facts and user search query.
        - Think about the starting point and the user search query.
        - Group the facts according to thier importance and relevance.
        - Make a topic for each group of facts.
        - Assing a color for each topic which gives simantic meaning for the topic. Stricktly use the colors from  ["#264653", "#27515B", "#287271", "#298880", "#2A988C"]. 
        - Organise the data facts in a way that it tells a cohesive story.
        - Make sure to maintain the flow of the story and make it engaging.
        
        Guidlines:
        - You can use the martini glass structure to organise the data facts in a cohesive way.
        - You are allowed to remove unwanted data facts.
        - The topic should be less than 40 characters.
        - You can maintain the supportive and contrasting data facts.
        - Do not change the details or formats in the given data. 
        
     
    """

    user_prompt = f"""
        #### 
        {paragraphs}
        ####
        
        ****
        {search_query}
        ****

    """

    console.print("[bold yellow]Creating Storyline...[/bold yellow]")
    vis_refined = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=StoryLine
    )
    console.print("[bold green]Creating Storyline Completed![/bold green]")
    return json.loads(vis_refined)


def find_shared_facts(clusters):
    shared_facts = []

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):

            common_facts = []
            cluster_i_facts = {fact["fact_id"]: fact for fact in clusters[i]["facts"]}
            cluster_j_facts = {fact["fact_id"]: fact for fact in clusters[j]["facts"]}

            common_fact_ids = set(cluster_i_facts.keys()) & set(cluster_j_facts.keys())

            for fact_id in common_fact_ids:
                common_facts.append(
                    {
                        "fact_id": fact_id,
                        "fact_content": cluster_i_facts[fact_id]["fact_content"],
                    }
                )

            if common_facts:
                shared_facts.append(
                    {
                        "start_cluster": clusters[i]["cluster_id"],
                        "end_cluster": clusters[j]["cluster_id"],
                        "count": len(common_facts),
                        "facts": common_facts,
                    }
                )

    return shared_facts


def find_shared_articles(clusters):
    shared_articles = []

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            common_articles = []
            cluster_i_articles = {
                article["article_id"]: article for article in clusters[i]["articles"]
            }
            cluster_j_articles = {
                article["article_id"]: article for article in clusters[j]["articles"]
            }

            common_article_ids = set(cluster_i_articles.keys()) & set(
                cluster_j_articles.keys()
            )

            for article_id in common_article_ids:
                common_articles.append(
                    {
                        "article_id": article_id,
                        "title": cluster_i_articles[article_id]["title"],
                        "url": cluster_i_articles[article_id]["url"],
                        "date": cluster_i_articles[article_id]["date"],
                        "year": cluster_i_articles[article_id]["year"],
                    }
                )

            if common_articles:
                shared_articles.append(
                    {
                        "start_cluster": clusters[i]["cluster_id"],
                        "end_cluster": clusters[j]["cluster_id"],
                        "count": len(common_articles),
                        "articles": common_articles,
                    }
                )

    return shared_articles


def destructure_id(fact_id):
    fact_id_parts = fact_id.split("_")
    article_id = int(fact_id_parts[0])
    article_fact_id = int(fact_id_parts[1])
    return article_id, article_fact_id


def get_article_dict(relatedness_results):
    article_dict = {
        str(item["article_meta_data"]["id"]): item["article_meta_data"]
        for item in relatedness_results["facts_with_meta"]
    }  # article_id: title, url, date

    for article in article_dict.values():
        article_date_str = article.get("date", "")
        year = ""
        date_format = DATE_CONFIGS["DATE_FORMAT"]  # Example: "19-01-2024"
        if article_date_str != "Unknown":
            parsed_date = convert_date(article_date_str)
            if parsed_date is not None:
                article_date = parsed_date.strftime(date_format)
                if parsed_date.year:
                    year = parsed_date.year
                    article["year"] = str(year)
        else:
            article["year"] = "Unknown"

    return article_dict


def check_shared_facts(clusters):
    shared_facts = []

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):

            common_facts = []
            cluster_i_facts = {
                fact["fact_id"]: fact for fact in clusters[i]["all_original_facts"]
            }
            cluster_j_facts = {
                fact["fact_id"]: fact for fact in clusters[j]["all_original_facts"]
            }

            common_fact_ids = set(cluster_i_facts.keys()) & set(cluster_j_facts.keys())

            for fact_id in common_fact_ids:
                common_facts.append(
                    {
                        "fact_id": fact_id,
                        "fact_content": cluster_i_facts[fact_id]["fact_content"],
                    }
                )

            if common_facts:
                shared_facts.append(
                    {
                        "start_cluster": clusters[i]["cluster_id"],
                        "end_cluster": clusters[j]["cluster_id"],
                        "count": len(common_facts),
                        "facts": common_facts,
                    }
                )

    return shared_facts


def check_shared_articles(clusters):
    shared_articles = []

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            common_articles = []
            cluster_i_articles = {
                article["id"]: article for article in clusters[i]["articles"]
            }
            cluster_j_articles = {
                article["id"]: article for article in clusters[j]["articles"]
            }

            common_article_ids = set(cluster_i_articles.keys()) & set(
                cluster_j_articles.keys()
            )

            for article_id in common_article_ids:
                common_articles.append(
                    {
                        "article_id": article_id,
                        "title": cluster_i_articles[article_id]["title"],
                        "url": cluster_i_articles[article_id]["url"],
                        "date": cluster_i_articles[article_id]["date"],
                        "year": cluster_i_articles[article_id]["year"],
                    }
                )

            if common_articles:
                shared_articles.append(
                    {
                        "start_cluster": clusters[i]["cluster_id"],
                        "end_cluster": clusters[j]["cluster_id"],
                        "count": len(common_articles),
                        "articles": common_articles,
                    }
                )

    return shared_articles


def calculate_word_frequencies(words, sentences):
    word_frequencies = defaultdict(int)
    words = [word.lower() for word in words]

    for sentence in sentences:
        sentence_words = re.findall(r"\b\w+\b", sentence.lower())
        for word in words:
            word_frequencies[word] += sentence_words.count(word)

    return dict(word_frequencies)


def get_article_frequencies_object(article_years):
    return [
        {"year": year, "number_of_articles": count}
        for year, count in Counter(article_years).items()
    ]


def calc_relatedness(query_embedding, item_embedding):
    relatedness_fn = lambda x, y: 1 - spatial.distance.cosine(x, y)
    return round(relatedness_fn(query_embedding, item_embedding), 2)


def get_score(fact, query_embedding):
    fact_embedding = gpt_helper.get_embeddings(fact["fact_content"])
    fact["relatedness_score"] = calc_relatedness(query_embedding, fact_embedding)
    return fact


def get_relatedness_scores(all_original_facts, query):

    query_embedding = gpt_helper.get_embeddings(query)

    with ThreadPoolExecutor() as executor:
        facts = list(
            executor.map(
                lambda fact: get_score(fact, query_embedding), all_original_facts
            )
        )

    return facts


def new_analyze_data(
    cluster_clickbait_list,
    detail_cluster_list,
    cluster_narrative_list,
    cluster_data,
    merged_facts,
    relatedness_results,
    search_query,
):

    cluster_detail_map = {
        topic["cluster_id"]: topic for topic in detail_cluster_list
    }  # cluster_id: title, description

    cluster_narratives_map = {
        cluster["cluster_id"]: cluster for cluster in cluster_narrative_list
    }  # cluster_id: facts (merged_id,merged_content,order_id,narrative)

    fact_dict = {
        fact["fact_id"]: fact
        for item in relatedness_results["facts_with_meta"]
        for fact in item["facts"]
    }  # fact_id: fact_content, fact_type, vis_data

    fact_group_dict = {
        fact_group["fact_group_id"]: fact_group
        for fact_group in cluster_data["all_fact_groups"]
    }  # fact_group_id: fact_group_content, fact_ids, article_ids, facts

    article_dict = get_article_dict(
        relatedness_results
    )  # article_id: title, url, date, year

    cluster_merged_facts = {cluster["cluster_id"]: cluster for cluster in merged_facts}
    # cluster_id: merged_facts (merged_id, merged_content, merged_data, merged_recommendation, titles, facts)
    # facts (fact_group_id, fact_group_content, fact_ids, article_ids)

    all_article_years = []
    all_merged_facts_in_order = []
    all_fact_groups_in_order = []
    all_original_facts_in_order = []

    all_unique_article_ids = set()

    for cluster in cluster_clickbait_list:

        cluster_id = cluster["cluster_id"]

        topic_info = cluster_detail_map.get(
            cluster_id, {}
        )  # cluster_id, title, description

        cluster["title"] = topic_info.get("title", "")
        cluster["description"] = topic_info.get("description", "")
        cluster["representative_facts"] = topic_info.get("representative_facts", "")
        cluster["cluster_order_id"] = topic_info.get("cluster_order_id", 0)

        cluster_narratives = cluster_narratives_map.get(
            cluster_id, {}
        )  # facts: merged_id,merged_content,order_id,narrative

        fact_narratives_dict = {
            fact["merged_id"]: fact for fact in cluster_narratives.get("facts", [])
        }

        merged_facts_data = cluster_merged_facts.get(
            str(cluster_id), {}
        )  # cluster_id, merged_facts

        unique_article_ids = set()
        years = []
        all_fact_groups = []
        all_original_facts = []
        all_fact_content = []
        articles = []

        for merged_fact in merged_facts_data.get("merged_facts", []):
            # merged_id, merged_content, merged_data, merged_recommendation, titles (chart_title, x_axis, y_axis), facts (fact_group_id,fact_group_content, fact_ids, article_ids)

            merged_id = merged_fact.get(
                "merged_id", ""
            )  # merged_id: {cluster_id}_{index}

            narrative_data = fact_narratives_dict.get(merged_id, {})
            order_id = narrative_data.get("order_id", 0)
            narrative = narrative_data.get("narrative", "")

            merged_fact["narrative"] = narrative
            merged_fact["order_id"] = order_id

            merged_fact_article_ids = set()
            merged_fact_group_ids = set()
            merged_articles = []

            fact_groups = merged_fact.get("facts", [])
            for fact_group in fact_groups:
                fact_group_id = fact_group.get("fact_group_id", "")
                merged_fact_group_ids.add(str(fact_group_id))
                fact_group_data = fact_group_dict.get(fact_group_id, {})

                article_ids = fact_group_data.get("article_ids", [])

                fact_group_articles = []

                for article_id in article_ids:
                    unique_article_ids.add(article_id)
                    all_unique_article_ids.add(article_id)
                    article = article_dict.get(str(article_id), {})
                    fact_group_articles.append(article)
                    merged_fact_article_ids.add(str(article_id))

                fact_group["fact_group_articles"] = fact_group_articles
                fact_group_facts = fact_group_data.get("facts", [])
                for fact in fact_group_facts:
                    fact_id = fact.get("fact_id", "")
                    id_data = fact_id.split("_")
                    article_id = id_data[0]
                    article = article_dict.get(article_id, {})
                    # if article.get("year", "") != "Unknown":
                    years.append(article.get("year", ""))
                    fact["article"] = article
                    all_original_facts.append(fact)
                    all_fact_content.append(fact.get("fact_content", ""))

                fact_group["fact_group_facts"] = fact_group_facts
                fact_group["number_of_similar_facts"] = len(fact_group_facts)
                all_fact_groups.append(fact_group)

            for article_id in merged_fact_article_ids:
                article = article_dict.get(str(article_id), {})
                merged_articles.append(article)

            merged_fact["merged_fact_article_ids"] = list(merged_fact_article_ids)
            merged_fact["merged_fact_group_ids"] = list(merged_fact_group_ids)
            merged_fact["merged_fact_article_count"] = len(merged_fact_article_ids)
            merged_fact["merged_fact_group_count"] = len(merged_fact_group_ids)
            merged_fact["merged_articles"] = merged_articles

        ordered_merged_facts = sorted(
            merged_facts_data.get("merged_facts", []), key=lambda x: x["order_id"]
        )
        cluster["merged_facts"] = ordered_merged_facts

        all_original_facts = get_relatedness_scores(all_original_facts, search_query)
        representative_facts = cluster["representative_facts"]
        representative_facts = get_relatedness_scores(
            representative_facts, search_query
        )

        cluster["number_of_merged_facts"] = len(cluster["merged_facts"])
        cluster["number_of_clickbaits"] = len(cluster.get("clickbait_list", []))
        cluster["all_fact_groups"] = all_fact_groups
        cluster["number_of_fact_groups"] = len(all_fact_groups)
        cluster["all_original_facts"] = all_original_facts
        cluster["number_of_original_facts"] = len(all_original_facts)
        cluster["number_of_articles"] = len(unique_article_ids)

        for art in unique_article_ids:
            articles.append(article_dict.get(str(art), {}))

        cluster["articles"] = articles
        if years:
            all_article_years.extend(years)
            clean_years = [year for year in years if year != "Unknown"]
            if clean_years:
                cluster["article_year_range"] = {
                    "earliest": min(clean_years),
                    "latest": max(clean_years),
                }
            else:
                cluster["article_year_range"] = {
                    "earliest": "Unknown",
                    "latest": "Unknown",
                }
        else:
            cluster["article_year_range"] = {"earliest": "Unknown", "latest": "Unknown"}

        important_word_frequency = calculate_word_frequencies(
            cluster["important_words"], all_fact_content
        )
        cluster["important_word_frequency"] = important_word_frequency
        cluster["all_fact_content"] = all_fact_content

    ordered_clusters = sorted(
        cluster_clickbait_list, key=lambda x: x["cluster_order_id"], reverse=True
    )

    step = {
        "cluster_id": 0,
        "start_step": 0,
        "end_step": -1,
    }

    steps = []
    max_original_facts = 0
    max_fact_groups = 0
    for index, cluster in enumerate(ordered_clusters):
        step["cluster_id"] = cluster["cluster_id"]
        step["start_step"] = step["end_step"] + 1
        step["end_step"] += cluster["number_of_merged_facts"]
        if max_original_facts < cluster["number_of_original_facts"]:
            max_original_facts = cluster["number_of_original_facts"]
        if max_fact_groups < cluster["number_of_fact_groups"]:
            max_fact_groups = cluster["number_of_fact_groups"]
        for entry in cluster["merged_facts"]:
            all_merged_facts_in_order.append(entry)
        for entry in cluster["all_fact_groups"]:
            all_fact_groups_in_order.append(entry)
        for entry in cluster["all_original_facts"]:
            all_original_facts_in_order.append(entry)
        steps.append(step.copy())

    all_clean_years = [year for year in all_article_years if year != "Unknown"]
    if all_clean_years:
        earliest_year = min(all_clean_years)
        latest_year = max(all_clean_years)
    else:
        earliest_year = "Unknown"
        latest_year = "Unknown"
    stats = {
        "total_articles": len(all_unique_article_ids),
        "total_clusters": len(ordered_clusters),
        "total_merged_facts": len(all_merged_facts_in_order),
        "total_fact_groups": len(all_fact_groups_in_order),
        "total_original_facts": len(all_original_facts_in_order),
        "max_original_facts": max_original_facts,
        "max_fact_groups": max_fact_groups,
        "article_year_range": {
            "earliest": earliest_year,
            "latest": latest_year,
        },
        "all_article_years": list(set(all_article_years)).sort(),
        "article_distribution": get_article_frequencies_object(all_article_years),
    }

    sorted_article_ids = sorted(
        int(article_id) for article_id in all_unique_article_ids
    )

    article_id_map = {
        article_id: index + 1 for index, article_id in enumerate(sorted_article_ids)
    }

    all_mapped_articles = []
    for article_id in sorted_article_ids:
        article = article_dict.get(str(article_id), {})
        all_mapped_articles.append(article)

    shared_facts = check_shared_facts(ordered_clusters)
    shared_articles = check_shared_articles(ordered_clusters)

    analysis_data = {
        "clusters": ordered_clusters,
        "stats": stats,
        "all_merged_facts_in_order": all_merged_facts_in_order,
        "all_fact_groups_in_order": all_fact_groups_in_order,
        "all_original_facts_in_order": all_original_facts_in_order,
        "steps": steps,
        "shared_facts": shared_facts,
        "shared_articles": shared_articles,
        "sorted_article_ids": article_id_map,
        "all_mapped_articles": all_mapped_articles,
    }

    return analysis_data


def fill_template(
    data, output_path="output.html", template_path=TEMPLATE_CONFIGS["TEMPLATE_PATH"]
):
    template = None
    with open(template_path, "r", encoding="utf-8") as file:
        template = file.read()
    data_str = json.dumps(data)
    # Parse the HTML with BeautifulSoup
    soup_ = BeautifulSoup(template, "html.parser")
    # Find the <script> tag with id "test"
    script_tag = soup_.find("script", id="jsonDataScript")
    script_tag.string = data_str

    temp_filename = output_path
    html_content = soup_.prettify()
    print(temp_filename)
    with open(temp_filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Make sure to use the absolute path
    absolute_path = os.path.abspath(temp_filename)

    # Open the HTML file in the default web browser
    webbrowser.open(f"file://{absolute_path}")
    return html_content, absolute_path


def assign_colors(merged_fact):
    THEME_CONFIGS = {"COLORS": ["#22d3ee", "#2dd4bf", "#f87171", "#facc15"]}
    colors = THEME_CONFIGS.get("COLORS", [])
    if not colors:
        colors = ["#808080"]  # Default gray if no colors available

    def hex_to_rgb(hex_color):
        """Convert hex color to RGB tuple with error handling"""
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 6:
            try:
                return (
                    int(hex_color[0:2], 16),
                    int(hex_color[2:4], 16),
                    int(hex_color[4:6], 16),
                )
            except ValueError:
                pass
        return (0, 0, 0)  # Return black for invalid colors

    for fact in merged_fact:
        try:
            narrative = fact.get("narrative", "")
            vis_data = fact.get("vis_data", [])
        except AttributeError:
            continue

        # Ensure valid types
        narrative = str(narrative) if isinstance(narrative, (str, int, float)) else ""
        vis_data = vis_data if isinstance(vis_data, list) else []

        for idx, data_point in enumerate(vis_data):
            if not isinstance(data_point, dict):
                continue

            # Get and escape values
            value = html.escape(str(data_point.get("value", "")))
            unit = html.escape(str(data_point.get("unit", "")))
            if not value:
                continue

            # Get color for this data point
            try:
                color_hex = colors[idx % len(colors)]
            except (TypeError, IndexError):
                color_hex = colors[0] if colors else "#808080"

            # Convert to RGB for background
            r, g, b = hex_to_rgb(color_hex)
            style = (
                f"color: {color_hex}; "
                f"background-color: rgba({r}, {g}, {b}, 0.1); "
                "font-weight: 600; padding: 0.25rem; border-radius: 0.25rem;"
            )

            # Try to replace value + unit combination first
            if unit:
                combined = f"{value} {unit}"
                if combined in narrative:
                    narrative = narrative.replace(
                        combined, f'<span style="{style}">{combined}</span>'
                    )
                    continue  # Skip to next data point after replacement

            # Replace value alone if combination not found
            narrative = narrative.replace(
                value, f'<span style="{style}">{value}</span>'
            )

        # Update the fact with modified narrative
        try:
            fact["narrative"] = narrative
        except TypeError:
            pass

    return merged_fact


def modify_recommendation(vis_recommendation, vis_data):
    # Process Pie Chart Cases
    if vis_recommendation == "pie":
        # Check if all units are the same
        units = [d["unit"].lower() for d in vis_data]
        if len(set(units)) != 1:
            vis_recommendation = "bar"
        else:
            unit = units[0]
            # Check if unit is either % or percentage
            if unit not in ["%", "percentage"]:
                vis_recommendation = "bar"
            else:
                # Calculate total value
                try:
                    total = sum(float(d["value"]) for d in vis_data)
                except ValueError:
                    total = 0  # In case of invalid values, consider total as 0 which will fail the check
                if total != 100:
                    vis_recommendation = "bar"

    # Process Bar Chart Cases
    if vis_recommendation == "bar":
        units = [d["unit"] for d in vis_data]
        if len(set(units)) > 1:
            # Count unit occurrences
            unit_counts = defaultdict(int)
            for u in units:
                unit_counts[u] += 1
            max_count = max(unit_counts.values())
            # Find all units with max count, select the first one in case of tie
            majority_units = [u for u, cnt in unit_counts.items() if cnt == max_count]
            selected_unit = majority_units[0]
            # Filter data points
            vis_data = [d for d in vis_data if d["unit"] == selected_unit]

    return {"vis_recommendation": vis_recommendation, "vis_data": vis_data}
