import itertools
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from common.utils import chunk_array, console, merge_arrays, write_to_json
from common.utils.timing_logger import LOGGER, log_execution_time
from stages.ArticleCrawler import collect_search_results
from stages.FactExtraction import (
    extract_and_filter_paragraphs,
    get_data_facts,
    get_data_values,
    refine_data,
    validate_data_extraction,
)
from stages.FactOrganization import (
    calculate_scores,
    clickbait_generation,
    cluster_detail_generation,
    cluster_facts,
    cluster_topics,
    correct_merged_facts,
    fill_missing_entities,
    get_entities_in_merged_facts,
    get_missing_entities,
    handle_filling_data,
    identify_similar_facts,
    merge_facts,
    organize_cluster_story,
    organize_data_by_topic,
    refine_cluster_detail,
    refine_merged_facts,
    relatedness,
    structure_fact_groups,
    structure_paragraphs,
    structure_paragraphs_with_meta_data,
    temporySimilarityCalculation,
    validate_merged_facts,
    vis_criticizer,
    vis_recommender,
    vis_refiner,
)
from stages.Presentation import (
    assign_colors,
    create_storyline,
    fill_template,
    format_overview,
    generate_wordcloud,
    modify_recommendation,
    new_analyze_data,
    refine_narrative,
    style_narrative,
)

threads = []


def check_fact_errors(data):
    for entry in data:
        for fact in entry.get("facts", []):
            if fact.get("error", []):
                return True
    return False


def process_data_validation(id, title, extracted_data):
    error_values = validate_data_extraction(id, title, extracted_data)
    return error_values["vis_data_error"]


def process_refine_data(id, title, validation_errors):
    refined_data = refine_data(id, title, validation_errors)
    return refined_data["data_facts_with_vis_data"]


# -------------- Start: Process Article  ----------
@log_execution_time
def process_article(
    id, title, date, link, article, search_query, file_path, iterations=1
):

    folder_path = f"{file_path}/{id}"

    title = title
    date = date
    url = link

    print_status(f"{id}: Started paragraph extraction")
    extracted_paragraphs = extract_and_filter_paragraphs(
        id, title, date, article, search_query
    )
    if not extracted_paragraphs or not extracted_paragraphs["paragraphs"]:
        return None
    extracted_paragraphs["id"] = id
    extracted_paragraphs["url"] = url
    extracted_paragraphs["title_original"] = title
    extracted_paragraphs["date_original"] = str(date)
    title = extracted_paragraphs["title"]
    date = extracted_paragraphs["date"]
    write_to_json(extracted_paragraphs, folder_path, "1_extracted_paragraphs.json")
    print_status(f"{id}: Finished paragraph extraction")

    # try:
    #     with open(f"{folder_path}/1_extracted_paragraphs.json", "r") as file:
    #         extracted_paragraphs = json.load(file)
    # except Exception as e:
    #     console.print(e)
    #     return None

    print_status(f"{id}: Started fact data extraction")
    data_facts_with_para = get_data_facts(id, title, extracted_paragraphs)
    if not data_facts_with_para or not data_facts_with_para["data_facts_with_para"]:
        return None
    write_to_json(data_facts_with_para, folder_path, "3_data_facts_with_para.json")
    print_status(f"{id}: Finished fact data extraction")

    # try:
    #     with open(f"{folder_path}/3_data_facts_with_para.json", "r") as file:
    #         data_facts_with_para = json.load(file)
    # except Exception as e:
    #     console.print(
    #         f"{folder_path}/3_data_facts_with_para.json file opening error", e
    #     )
    #     return None

    print_status(f"{id}: Started data value extraction")
    data_fact_with_vis_data = get_data_values(
        id, title, date, data_facts_with_para, article
    )
    if (
        not data_fact_with_vis_data
        or not data_fact_with_vis_data["data_facts_with_vis_data"]
    ):
        return None
    write_to_json(
        data_fact_with_vis_data, folder_path, "4_data_fact_with_vis_data.json"
    )
    print_status(f"{id}: Finished data value extraction")

    # try:
    #     with open(f"{folder_path}/4_data_fact_with_vis_data.json", "r") as file:
    #         data_fact_with_vis_data = json.load(file)
    # except Exception as e:
    #     console.print(e)
    #     return None

    print_status(f"{id}: Started first data validation")
    # validation_errors = validate_data_extraction(id, title, data_fact_with_vis_data)
    with ThreadPoolExecutor() as executor:
        validation_errors_nested = list(
            executor.map(
                process_data_validation,
                itertools.repeat(id),
                itertools.repeat(title),
                data_fact_with_vis_data["data_facts_with_vis_data"],
            )
        )

    if not validation_errors_nested:
        return None

    error_list = list(itertools.chain.from_iterable(validation_errors_nested))
    validation_errors = {
        "has_error": check_fact_errors(error_list),
        "vis_data_error": error_list,
    }

    write_to_json(validation_errors, folder_path, "5_validation_errors.json")
    print_status(f"{id}: Finished first data validation")

    # try:
    #     with open(f"{folder_path}/5_validation_errors.json", "r") as file:
    #         validation_errors = json.load(file)
    # except Exception as e:
    #     console.print(e)
    #     return None

    i = 0
    print_status(f"{id}: Started iterative validation")
    while i < iterations and validation_errors["has_error"]:
        # refined_data = refine_data(id, title, validation_errors)
        with ThreadPoolExecutor() as executor:
            refined_data_nested = list(
                executor.map(
                    process_refine_data,
                    itertools.repeat(id),
                    itertools.repeat(title),
                    validation_errors["vis_data_error"],
                )
            )

        all_refined_data = list(itertools.chain.from_iterable(refined_data_nested))
        refined_data = {
            "data_facts_with_vis_data": all_refined_data,
        }
        if not refined_data or not refined_data["data_facts_with_vis_data"]:
            break

        data_fact_with_vis_data = refined_data

        write_to_json(
            data_fact_with_vis_data, folder_path, f"6_refined_data_{i+1}.json"
        )

        if iterations - i > 1:
            # validation_errors = validate_data_extraction(
            #     id, title, data_fact_with_vis_data
            # )
            with ThreadPoolExecutor() as executor:
                validation_errors_nested = list(
                    executor.map(
                        process_data_validation,
                        itertools.repeat(id),
                        itertools.repeat(title),
                        data_fact_with_vis_data["data_facts_with_vis_data"],
                    )
                )

            if not validation_errors_nested:
                return None

            error_list = list(itertools.chain.from_iterable(validation_errors_nested))
            validation_errors = {
                "has_error": check_fact_errors(error_list),
                "vis_data_error": error_list,
            }
            write_to_json(
                validation_errors, folder_path, f"7_validation_errors_{i+1}.json"
            )
        i += 1

    if (
        not data_fact_with_vis_data
        or not data_fact_with_vis_data["data_facts_with_vis_data"]
    ):
        return None
    write_to_json(
        data_fact_with_vis_data, folder_path, "8_data_fact_with_vis_data_final.json"
    )
    print_status(f"{id}: Finished iterative validation")

    # try:
    #     with open(f"{folder_path}/8_data_fact_with_vis_data_final.json", "r") as file:
    #         data_fact_with_vis_data = json.load(file)
    # except Exception as e:
    #     console.print(e)
    #     return None

    print_status(f"{id}: Started structuring data")
    structured_data = structure_paragraphs_with_meta_data(
        id, title, date, link, data_fact_with_vis_data
    )
    if not structured_data or not structured_data["data_facts_with_vis_data_meta"]:
        return None
    write_to_json(structured_data, folder_path, "9_structred_data.json")
    print_status(f"{id}: Finished structuring data")

    return structured_data


# -------------- End: Process Article  ----------


# Function to process each article in a thread
@log_execution_time
def process_article_thread(
    id, title, date, link, article, search_query, results, file_path, iterations=1
):
    try:
        data_with_meta = process_article(
            id,
            title,
            date,
            link,
            article,
            search_query,
            file_path,
            iterations,
        )

        if not data_with_meta or not data_with_meta["data_facts_with_vis_data_meta"]:
            return None

        results["facts_with_meta"].extend(
            data_with_meta["data_facts_with_vis_data_meta"]
        )
        results["all_paragraphs"].extend(data_with_meta["all_paragraphs"])
        results["all_facts"].extend(data_with_meta["all_facts"])
        results["all_facts_with_vis_data"].extend(
            data_with_meta["all_facts_with_vis_data"]
        )

        return results
    except Exception as e:
        LOGGER.error(f"Error processing article:{id}, {title}, {e}")
        console.print(f"Error processing article:{id}, {title}, {e}")


def run_clickbait_and_detail_generation(clusters, search_query):
    with ThreadPoolExecutor() as executor:
        futures = [
            (
                executor.submit(clickbait_generation, cluster, search_query),
                executor.submit(cluster_detail_generation, cluster, search_query),
            )
            for cluster in clusters
        ]

    cluster_clickbait_list = []
    detail_cluster_list = []
    for future_clickbait, future_detail in futures:
        cluster_clickbait_list.append(future_clickbait.result())
        detail_cluster_list.append(future_detail.result())
    return cluster_clickbait_list, detail_cluster_list


def print_status(status):
    status_log = f"STATE:[bold yellow] {status} [/bold yellow]"
    LOGGER.info(status_log)
    console.print(status_log)


def process_similar_facts(cluster):
    return identify_similar_facts(cluster)


def process_fact_narrative(fact):
    fact_narrative = {
        "narrative": fact["narrative"],
        "vis_recommendation": fact["merged_recommendation"],
        "vis_data": fact["merged_data"],
    }
    styled_narrative = style_narrative(fact_narrative)
    fact["narrative"] = styled_narrative["narrative"]
    fact["vis_recommendation"] = styled_narrative["vis_recommendation"]
    fact["vis_data"] = styled_narrative["vis_data"]
    return fact


def process_refine_narrative(fact):
    fact_narrative = {
        "narrative": fact["narrative"],
        "vis_recommendation": fact["merged_recommendation"],
        "vis_data": fact["merged_data"],
    }
    styled_narrative = refine_narrative(fact_narrative)
    fact["narrative"] = styled_narrative["narrative"]
    fact["vis_recommendation"] = styled_narrative["vis_recommendation"]
    fact["vis_data"] = styled_narrative["vis_data"]
    return fact


def process_assign_colors(fact):
    fact_narrative = {
        "narrative": fact["narrative"],
        "vis_recommendation": fact["merged_recommendation"],
        "vis_data": fact["merged_data"],
    }
    styled_narrative = assign_colors(fact_narrative)
    fact["narrative"] = styled_narrative["narrative"]
    fact["vis_recommendation"] = styled_narrative["vis_recommendation"]
    fact["vis_data"] = styled_narrative["vis_data"]
    return fact


def process_merge_facts(fact_group):
    return merge_facts(fact_group)


def process_validate_merged_facts(merged_facts):
    errors = validate_merged_facts(merged_facts)
    merged_facts["errors"] = errors["errors"]
    return merged_facts


def process_refine_merged_facts(merged_facts):
    return refine_merged_facts(merged_facts)


def process_correct_merged_facts(merged_facts):
    return correct_merged_facts(merged_facts)


def get_culster_merged_facts(merged_facts):

    for cluster in merged_facts:
        new_cluster = {"cluster_id": cluster["cluster_id"]}

    return cluster_topics(merged_facts)


def get_merged_clusters(merged_facts):
    new_clusters = []
    for cluster in merged_facts:
        new_cluster = {"cluster_id": cluster["cluster_id"]}
        new_merged_facts = []
        for merged_fact in cluster["merged_facts"]:
            new_merged_fact = {
                "merged_id": merged_fact["merged_id"],
                "merged_content": merged_fact["merged_content"],
            }
            new_merged_facts.append(new_merged_fact)
        new_cluster["facts"] = new_merged_facts
        new_clusters.append(new_cluster)
    return new_clusters


def process_organizing_story(cluster):
    return organize_cluster_story(cluster)


def run_refine_detail_and_organize_story(
    detail_cluster_list, filtered_merged_clusters, search_query
):
    with ThreadPoolExecutor() as executor:
        refine_future = executor.submit(
            refine_cluster_detail, detail_cluster_list, search_query
        )
        narrative_future = executor.submit(
            lambda: list(
                executor.map(process_organizing_story, filtered_merged_clusters)
            )
        )

        refine_result = None
        narrative_result = None

        for future in as_completed([refine_future, narrative_future]):
            if future == refine_future:
                refine_result = future.result()
                refine_result = refine_result["clusters"]
            elif future == narrative_future:
                narrative_result = future.result()

        return refine_result, narrative_result


def process_merged_fact_entity_recognition(merged_fact):
    return get_entities_in_merged_facts(merged_fact)


def process_wordcloud_generation(analysis, cluster_results):

    cluster_wise_facts = cluster_results["cluster_wise_facts"]
    cluster_summaries = cluster_results["cluster_summary"]
    clusters = analysis["clusters"]
    stats = analysis["stats"]
    max_riginal_facts = stats["max_original_facts"]

    def get_wordcloud(cluster, cluster_wise_fact):
        return generate_wordcloud(cluster, max_riginal_facts, cluster_wise_fact)

    with ThreadPoolExecutor() as executor:
        new_clusters = list(executor.map(get_wordcloud, clusters, cluster_wise_facts))

    for new_cluster in new_clusters:
        cluster_id = new_cluster["cluster_id"]
        new_cluster["cluster_summary"] = cluster_summaries[str(cluster_id)]

    analysis["clusters"] = new_clusters
    return analysis


def process_cluster_entity_recognition(merged_facts_data):
    def process_cluster_merged_facts(cluster):
        with ThreadPoolExecutor() as executor:
            # Process merged facts within each cluster in parallel
            processed_facts = list(
                executor.map(
                    process_merged_fact_entity_recognition, cluster["merged_facts"]
                )
            )
        # Replace the original merged facts with processed facts
        cluster["merged_facts"] = processed_facts
        return cluster

    # Process each cluster's merged facts in parallel
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_cluster_merged_facts, merged_facts_data))

    return results


def find_missing_fact_groups(fact_groups, merged_facts):
    missing_fact_groups_per_cluster = []

    # Organize fact_groups by cluster_id
    fact_groups_dict = {
        cluster["cluster_id"]: {
            group["fact_group_id"]: group for group in cluster["fact_groups"]
        }
        for cluster in fact_groups
    }

    # Organize merged_facts by cluster_id
    merged_facts_dict = {cluster["cluster_id"]: set() for cluster in merged_facts}
    for cluster in merged_facts:
        for merged in cluster["merged_facts"]:
            for fact in merged["facts"]:
                merged_facts_dict[cluster["cluster_id"]].add(fact["fact_group_id"])

    # Find missing fact groups per cluster
    for cluster_id, group_dict in fact_groups_dict.items():
        merged_ids = merged_facts_dict.get(cluster_id, set())
        missing_groups = [
            group
            for group_id, group in group_dict.items()
            if group_id not in merged_ids
        ]
        if missing_groups:
            missing_fact_groups_per_cluster.append(
                {"cluster_id": cluster_id, "fact_groups": missing_groups}
            )

    return missing_fact_groups_per_cluster


def combine_merged_results(original_clusters, missing_clusters):
    missing_clusters_dict = {
        cluster["cluster_id"]: cluster["merged_facts"] for cluster in missing_clusters
    }

    for cluster in original_clusters:
        cluster_id = cluster["cluster_id"]
        missing_facts = missing_clusters_dict.get(cluster_id, [])
        cluster["merged_facts"].extend(missing_facts)

    for cluster in original_clusters:
        cluster_id = cluster["cluster_id"]
        for index, merged in enumerate(cluster["merged_facts"]):
            merged["merged_id"] = f"{cluster_id}_{index}"

    return original_clusters


def process_fact_grouping(clusters):
    with ThreadPoolExecutor() as executor:
        fact_groups = list(
            executor.map(process_similar_facts, clusters["cluster_wise_facts"])
        )
    return fact_groups


def process_merging_facts(cluster_data):
    with ThreadPoolExecutor() as executor:
        merged_facts = list(executor.map(process_merge_facts, cluster_data["clusters"]))
    return merged_facts


def refine_missing_entities(missing_entities_evaluated):
    with ThreadPoolExecutor() as executor:
        return list(
            executor.map(process_refine_merged_facts, missing_entities_evaluated)
        )


def run_merged_facts_validation(merged_facts):
    with ThreadPoolExecutor() as executor:
        return list(executor.map(process_validate_merged_facts, merged_facts))


def run_correcting_merged_facts(validated_facts):
    with ThreadPoolExecutor() as executor:
        corrected_merged_facts = list(
            executor.map(process_correct_merged_facts, validated_facts)
        )
    return corrected_merged_facts


def run_refine_all_facts_in_order(all_facts_in_order):
    with ThreadPoolExecutor() as executor:
        return list(executor.map(process_refine_narrative, all_facts_in_order))


def add_meta_data(articles, data):

    titles = data["Title"].values
    meta_descriptions = data["Meta_Description"].values
    favicons = data["Favicon"].values
    sources = data["Source"].values
    domains = data["Domain"].values
    displayed_links = data["Displayed_Link"].values
    snippet_highlighted_words = data["Snippet_Highlighted_Words"].values

    for article in articles:
        article["result_title"] = titles[article["id"]]
        article["meta_description"] = meta_descriptions[article["id"]]
        article["favicon"] = favicons[article["id"]]
        article["source"] = sources[article["id"]]
        article["domain"] = domains[article["id"]]
        article["displayed_link"] = displayed_links[article["id"]]
        article["snippet_highlighted_words"] = snippet_highlighted_words[article["id"]]
    return articles


def add_cluster_similarities(clusters, cluster_summary):
    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        cluster["cluster_similarity_score"] = cluster_summary.get(str(cluster_id), {})[
            "mean_similarity_normalized"
        ]
        cluster["top_keywords"] = cluster_summary.get(str(cluster_id), {})[
            "top_keywords"
        ]
    return clusters


def run_style_narratives(facts_list):
    with ThreadPoolExecutor() as executor:
        styled_facts = list(executor.map(process_fact_narrative, facts_list))
    return styled_facts


def process_modify_recommendation(fact):
    modified_data = modify_recommendation(fact["vis_recommendation"], fact["vis_data"])
    fact["vis_recommendation"] = modified_data["vis_recommendation"]
    fact["vis_data"] = modified_data["vis_data"]
    return fact


def run_modify_recommendation(facts_list):
    with ThreadPoolExecutor() as executor:
        styled_facts = list(executor.map(process_assign_colors, facts_list))
    return styled_facts


@log_execution_time
def generate_story(
    search_query,
    web="pewresearch.org",
    result_count_per_page=1,
    iterations=1,
    search_result_file="google_search_pew.csv",
    output_path="story.html",
    results_path="JsonOutputs",
    country_code="sg",
    num_pages=2,
):
    try:
        file_path = results_path

        print_status("Started collecting search results")
        collect_search_results(
            search_query, web, num_results=result_count_per_page, num_pages=num_pages, csv_filename=search_result_file, country_code=country_code
        )
        print_status("Finished collecting search results")
        

        df = pd.read_csv(search_result_file)
        articles = df["Page_Content"].values
        meta_descriptions = df["Meta_Description"].values
        links = df["Link"].values
        dates = df["Date"].values
        titles = df["Title"].values
        ids = df["id"].values
        headings = df["Headings"].values

        results = {
            "facts_with_meta": [],
            "all_paragraphs": [],
            "all_facts": [],
            "all_facts_with_vis_data": [],
        }

        print_status("Started running article processing threads")

        @log_execution_time
        def process_articles(id):
            process_article_thread(
                id,
                titles[id],
                dates[id],
                links[id],
                articles[id],
                search_query,
                results,
                file_path,
                iterations,
            )

        with ThreadPoolExecutor() as executor:
            executor.map(process_articles, range(len(articles)))
        write_to_json(results, file_path, "1_extracted_data.json")
        print_status("Finished article processing threads")

        with open(f"{file_path}/1_extracted_data.json", "r") as file:
            results = json.load(file)

        print_status("Started clustering facts")
        clusters = cluster_facts(results["all_facts"], file_path, search_query)
        write_to_json(clusters, file_path, "2_clusters.json")
        print_status("Finished clustering facts")

        with open(f"{file_path}/2_clusters.json", "r") as file:
            clusters = json.load(file)

        # TODO - Remove this later
        # cluster_summaries = temporySimilarityCalculation(
        #     clusters, "Global cancer patients burden"
        # )
        # write_to_json(cluster_summaries, file_path, "2_clusters.json")

        print_status("Started fact similarity check")
        fact_groups = process_fact_grouping(clusters)
        write_to_json(fact_groups, file_path, "3_fact_groups.json")
        print_status("Finished fact similarity check")

        with open(f"{file_path}/3_fact_groups.json", "r") as file:
            fact_groups = json.load(file)

        print_status("Started structuring fact groups")
        cluster_data = structure_fact_groups(fact_groups, results)
        write_to_json(cluster_data, file_path, "4_cluster_wise_fact_groups.json")
        print_status("Finished structuring fact groups")

        with open(f"{file_path}/4_cluster_wise_fact_groups.json", "r") as file:
            cluster_data = json.load(file)

        print_status("Started merging facts")
        merged_facts = process_merging_facts(cluster_data)
        write_to_json(merged_facts, file_path, "5_merged_facts.json")
        print_status("Finished merging facts")

        with open(f"{file_path}/5_merged_facts.json", "r") as file:
            merged_facts = json.load(file)

        print_status("Started identifying missing entities")
        missing_entities = get_missing_entities(merged_facts)
        write_to_json(missing_entities, file_path, "6_missing_entities.json")
        print_status("Finished identifying missing entities")

        with open(f"{file_path}/6_missing_entities.json", "r") as file:
            missing_entities = json.load(file)

        print_status("Started filling missing entities")
        filled_entities = handle_filling_data(missing_entities, articles)
        write_to_json(filled_entities, file_path, "7_filled_entities.json")
        print_status("Finished filling missing entities")

        with open(f"{file_path}/7_filled_entities.json", "r") as file:
            filled_entities = json.load(file)

        print_status("Started identifying missing entities for filled data")
        missing_entities_evaluated = get_missing_entities(filled_entities)
        write_to_json(
            missing_entities_evaluated, file_path, "8_missing_entities_after_fill.json"
        )
        print_status("Finished identifying missing entities for filled data")

        with open(f"{file_path}/8_missing_entities_after_fill.json", "r") as file:
            missing_entities_evaluated = json.load(file)

        print_status("Started refining merged facts")
        merged_facts = refine_missing_entities(missing_entities_evaluated)
        write_to_json(merged_facts, file_path, "9_refined_merged_facts.json")
        print_status("Finished refining merged facts")

        with open(f"{file_path}/9_refined_merged_facts.json", "r") as file:
            merged_facts = json.load(file)

        print_status("Started validating merged facts")
        validated_facts = run_merged_facts_validation(merged_facts)
        write_to_json(validated_facts, file_path, "10_validated_facts.json")
        print_status("Finished validating merged facts")

        with open(f"{file_path}/10_validated_facts.json", "r") as file:
            validated_facts = json.load(file)

        print_status("Started correcting merged facts")
        merged_facts = run_correcting_merged_facts(validated_facts)
        write_to_json(merged_facts, file_path, "11_corrected_merged_facts.json")
        print_status("Finished correcting merged facts")

        with open(f"{file_path}/11_corrected_merged_facts.json", "r") as file:
            merged_facts = json.load(file)

        ###  For narrative generation
        print_status("Started filtering merged clusters")
        filtered_merged_clusters = get_merged_clusters(merged_facts)
        write_to_json(
            filtered_merged_clusters, file_path, "12_filtered_merged_clusters.json"
        )
        print_status("Finished filtering merged clusters")

        with open(f"{file_path}/12_filtered_merged_clusters.json", "r") as file:
            filtered_merged_clusters = json.load(file)

        print_status("Started detailing clusters and clickbait generation")
        cluster_clickbait_list, detail_cluster_list = (
            run_clickbait_and_detail_generation(
                cluster_data["cluster_wise_facts"], search_query
            )
        )
        write_to_json(
            cluster_clickbait_list, file_path, "13_cluster_clickbait_list.json"
        )
        write_to_json(detail_cluster_list, file_path, "14_detail_cluster_list.json")
        print_status("Finished detailing clusters")

        with open(f"{file_path}/14_detail_cluster_list.json", "r") as file:
            detail_cluster_list = json.load(file)

        print_status("Started refining detailed clusters and organizing story")
        detail_cluster_list, cluster_narrative_list = (
            run_refine_detail_and_organize_story(
                detail_cluster_list, filtered_merged_clusters, search_query
            )
        )
        write_to_json(
            detail_cluster_list, file_path, "15_refined_detail_cluster_list.json"
        )
        write_to_json(
            cluster_narrative_list, file_path, "16_cluster_narrative_list.json"
        )
        print_status("Finished refining detailed clusters and organizing story")

        with open(f"{file_path}/13_cluster_clickbait_list.json", "r") as file:
            cluster_clickbait_list = json.load(file)

        with open(f"{file_path}/15_refined_detail_cluster_list.json", "r") as file:
            detail_cluster_list = json.load(file)

        with open(f"{file_path}/16_cluster_narrative_list.json", "r") as file:
            cluster_narrative_list = json.load(file)

        print_status("Started analysing data")
        analysis = new_analyze_data(
            cluster_clickbait_list,
            detail_cluster_list,
            cluster_narrative_list,
            cluster_data,
            merged_facts,
            results,
            search_query,
        )
        with open(f"{file_path.rsplit("/", 1)[0]}/queries.json", "r") as file:
            queries = json.load(file)
        analysis["search_queries"] = queries["search_queries"]
        write_to_json(analysis, file_path, "17_analysis.json")
        print_status("Finished detailing clusters")

        with open(f"{file_path}/17_analysis.json", "r") as file:
            analysis = json.load(file)

        print_status("Started narrative styling")
        all_facts_in_order = analysis["all_merged_facts_in_order"]
        with ThreadPoolExecutor() as executor:
            all_facts_in_order = list(
                executor.map(process_fact_narrative, all_facts_in_order)
            )
        analysis["all_merged_facts_in_order"] = all_facts_in_order
        write_to_json(analysis, file_path, "18_styled_analysis.json")

        with open(f"{file_path}/18_styled_analysis.json", "r") as file:
            analysis = json.load(file)

        print_status("Started narrative styling")
        all_facts_in_order = analysis["all_merged_facts_in_order"]
        all_facts_in_order = run_refine_all_facts_in_order(all_facts_in_order)
        analysis["all_merged_facts_in_order"] = all_facts_in_order
        write_to_json(analysis, file_path, "19_refined_styled_analysis.json")
        print_status("Finished narrative styling")

        with open(f"{file_path}/19_refined_styled_analysis.json", "r") as file:
            analysis = json.load(file)

        print_status("Started adding article meta data")
        analysis["all_mapped_articles"] = add_meta_data(
            analysis["all_mapped_articles"], df
        )

        analysis["clusters"] = add_cluster_similarities(
            analysis["clusters"], clusters["cluster_summary"]
        )
        # analysis = process_wordcloud_generation(analysis, clusters)
        write_to_json(analysis, file_path, "20_final_styled_analysis.json")
        print_status("Finished adding article meta data")

        return analysis

    except Exception as e:
        LOGGER.error(f"Error: {e}")
        console.print(f"Error: {e}")
        return None
    finally:
        console.print("finished")
