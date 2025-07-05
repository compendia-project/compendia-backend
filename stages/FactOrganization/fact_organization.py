import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List

from common.config import PROMPTS
from common.gpt_helper import GPTHelper
from common.utils import console
from common.utils.timing_logger import LOGGER, log_execution_time
from fastapi.encoders import jsonable_encoder
from models.models import (
    ArticleDataFactVisDataMeta,
    ArticleVisRecommendation,
    ArticleVisRecommendationFeedback,
    ClusterClickbait,
    ClusterNarrative,
    Clusters,
    DataFactWithMetaData,
    DataFactWithRelatedness,
    DetailCluster,
    DetailClusters,
    Errors,
    FactGroups,
    FactGroupWithArticleData,
    FactGroupWithMissingEntity,
    MergedFactEntities,
    MergedFacts,
    MergedFactsEntities,
)
from scipy import spatial

gpt_helper = GPTHelper()


@log_execution_time
def structure_paragraphs(refined_data, article, link):
    system_prompt = """
        You are an expert assistant specialized in organizing content with metadata.
        
        **Task:**
        You will be provided with an original document (between `####` delimiters) and extracted content (between `****` delimiters). 
        Your task is to extract the metadata from the original document and then structure the extracted content using this metadata.
        Link is given between `$$$$` delimiters.

        **Guidelines:**
        - **Extraction:** Identify and extract key metadata from the original document, such as Title, Date published, and any other relevant information.
        - **Structuring:** Use the extracted metadata to organize the extracted content logically and coherently.
        - **Formatting:** Follow the output format precisely, ensuring clarity and readability.
        - **Verification:** Double-check the extracted metadata and content for accuracy and completeness.

        **Let's think step by step:**
        1. **Read** the original document provided between `####` delimiters.
        2. **Extract** all relevant metadata from the document.
        4. **Structure** the extracted content with the metadata.
        5. **Format** the final output as per the required structure.
        6. **Review** the output to ensure all guidelines are met.
    """

    user_prompt = f"""
    ####
    {article}
    ####
    
    ****
    {refined_data}
    **** 
    
    $$$$
    {link}
    $$$$

    """
    console.print(
        "[bold yellow]Structuring the paragraph with meta data...[/bold yellow]"
    )
    structre_paragraphs = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=ArticleDataFactVisDataMeta
    )

    if structre_paragraphs is None:
        console.print("[bold red]Error: Structuring failed.[/bold red]")
        LOGGER.error("Error: Structuring failed.")
        return None

    try:
        obj = json.loads(structre_paragraphs)
        console.print(
            "[bold green]Structuring the paragraph with meta data Completed![/bold green]"
        )
        return obj
    except json.JSONDecodeError:
        console.print(
            "[bold red]Error: Failed to parse the response into JSON.[/bold red]"
        )
        LOGGER.error("Error: Failed to parse the response into JSON.")
        return None


def structure_paragraphs_with_meta_data(id, title, date, url, refined_data):
    try:
        total_facts = 0
        all_paragraphs = []
        all_facts = []
        all_facts_with_vis_data = []
        for index, item in enumerate(refined_data["data_facts_with_vis_data"]):
            item["article_meta_data"] = {
                "title": title,
                "date": str(date),
                "url": url,
                "id": id,
            }
            item["para_id"] = f"{id}_{index}"
            all_paragraphs.append(
                {"para_id": item["para_id"], "paragraph": item["paragraph"]}
            )
            for fact in item["facts"]:
                fact["fact_id"] = f"{id}_{index}_{total_facts}"
                total_facts += 1
                all_facts.append(
                    {"fact_id": fact["fact_id"], "fact_content": fact["fact_content"]}
                )
                all_facts_with_vis_data.append(fact)

        new_data = {
            "data_facts_with_vis_data_meta": refined_data["data_facts_with_vis_data"],
            "all_paragraphs": all_paragraphs,
            "all_facts": all_facts,
            "all_facts_with_vis_data": all_facts_with_vis_data,
        }
        return new_data

    except Exception as e:
        LOGGER.error(e)
        return None


def calculate_relatedness(query_embedding, paragraph):
    relatedness_fn = lambda x, y: 1 - spatial.distance.cosine(x, y)
    para_embedding = gpt_helper.get_embeddings(paragraph)
    return relatedness_fn(query_embedding, para_embedding)


def calc_relatedness(query_embedding, item_embedding):
    relatedness_fn = lambda x, y: 1 - spatial.distance.cosine(x, y)
    return relatedness_fn(query_embedding, item_embedding)


def process_fact(fact, query_embedding):
    try:
        fact_embedding = gpt_helper.get_embeddings(fact["fact_content"])
        fact["fact_score"] = calc_relatedness(query_embedding, fact_embedding)
        return fact
    except Exception as e:
        console.print("[bold red]Error: Failed to process fact.[/bold red]")
        LOGGER.error(e)
        return None


def process_item(item, query_embedding):
    try:
        paragraph_embedding = gpt_helper.get_embeddings(item["paragraph"])
        item["paragraph_score"] = calc_relatedness(query_embedding, paragraph_embedding)

        # Process facts in parallel
        with ThreadPoolExecutor() as executor:
            item["facts"] = list(
                executor.map(
                    lambda fact: process_fact(fact, query_embedding), item["facts"]
                )
            )

        return item
    except Exception as e:
        console.print("[bold red]Error: Failed to process item.[/bold red]")
        LOGGER.error(e)
        return None


@log_execution_time
def calculate_scores(query, data):
    try:
        query_embedding = gpt_helper.get_embeddings(query)

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(
                    lambda item: process_item(item, query_embedding),
                    data,
                )
            )

        return results
    except Exception as e:
        console.print("[bold red]Error: Failed to calculate scores.[/bold red]")
        LOGGER.error(e)
        return None


@log_execution_time
def relatedness(query, paragraphs: List[DataFactWithMetaData]):
    query_embedding = gpt_helper.get_embeddings(query)

    def process_paragraph(para):
        relatedness_score = calculate_relatedness(query_embedding, para["paragraph"])
        if relatedness_score > 0.3:
            return DataFactWithRelatedness(
                **para,
                relatedness_score=relatedness_score,
            )
        return None

    with ThreadPoolExecutor() as executor:
        # Process paragraphs in parallel and collect results
        results = list(executor.map(process_paragraph, paragraphs))

    # Filter out None values and sort
    para_with_relatedness = [result for result in results if result is not None]
    sorted_data_facts = sorted(
        para_with_relatedness, key=lambda x: x.relatedness_score, reverse=True
    )

    sorted_data_facts = jsonable_encoder(sorted_data_facts)
    return sorted_data_facts


@log_execution_time
def vis_recommender(paragraphs):
    system_prompt = """
        You are an excellent assistant in visualization recommendation.
        **Task:**
        Given a list of paragraphs and related data facts (between #### delimiters), your task is to recommend the most suitable visualization type for the facts.

        **Guidelines:**
        - Carefully read through the provided paragraphs and data facts.
        - Identify key information and data that can be visualized.
        - Check the extracted data values provided.
        - Recommend the most suitable visualization type that effectively represents the data and enhances understanding.
        - Here is the list of visulization types
          - bar
          - line
          - isotype
          - map
          - scatter_plot
          - pie
          - area 
          - bubble
          - text
          - table
          - box_plot
          - treemap
        - Reorganize the data values in a way that is suitable for the recommended visualization type. For example, order the data values according to the timeline. 
        - Sort the data values if they are time series data.

        **Let's think step by step:**
        1. **Read the Paragraphs and Data Facts:**
        - Thoroughly read the content between the #### delimiters.
        2. **Identify Visualizable Elements:**
        - Extract key data points, trends, comparisons, or relationships mentioned.
        3. **Analyze the Data Values:**
        - Examine the extracted data for variables, categories, time series, or correlations.
        4. **Determine Suitable Visualization Types:**
        - Consider charts like bar graphs, line charts, scatter plots, pie charts, etc.
        - Choose the type that best represents the data and conveys insights effectively.
        5. **Justify Your Recommendation:**
        - Explain why the chosen visualization is appropriate for the data.
        6. **Reorganize Data Values:**
        - Restructure the data values to align with the recommended visualization type considering the order, grouping, or formatting.
        
        **Example:**
        
        _Input:_
        {
            "paragraph": "In late March, views of how Trump was handling the outbreak were already starkly split along party lines. Around eight-in-ten Republicans (83%) said the president was doing an excellent or good job, including 47% who said he was doing an excellent job. A nearly identical share of Democrats (81%) rated his response as only fair or poor, including 56% who said it was poor.",
            "facts": [
                {
                    "fact_type": "proportion",
                    "fact_content": "83% of Republicans said the president was doing an excellent or good job."
                },
                {
                    "fact_type": "proportion",
                    "fact_content": "47% of Republicans said he was doing an excellent job."
                },
                {
                    "fact_type": "proportion",
                    "fact_content": "81% of Democrats rated his response as only fair or poor."
                },
                {
                    "fact_type": "proportion",
                    "fact_content": "56% of Democrats said it was poor."
                }
            ],
            "vis_data": [
                {
                    "x": "Republicans (Excellent/Good)",
                    "y": "83%",
                    "color": "#ff4d4d"
                },
                {
                    "x": "Republicans (Excellent)",
                    "y": "47%",
                    "color": "#ff5050"
                },
                {
                    "x": "Democrats (Fair/Poor)",
                    "y": "81%",
                    "color": "#009933"
                },
                {
                    "x": "Democrats (Poor)",
                    "y": "56%",
                    "color": "#53c653"
                }
            ],
            "titles": {
                "chart_title": "Partisan Views on Trump's COVID-19 Response (March 2020)",
                "x_axis": "Political Affiliation",
                "y_axis": "Percentage"
            },
            "article_meta_data": {
                "title": "A Year of U.S. Public Opinion on the Coronavirus Pandemic",
                "date": "March 5, 2021"
            }
        }
        
        _Output:_
        
        {
            "paragraph": "In late March, views of how Trump was handling the outbreak were already starkly split along party lines. Around eight-in-ten Republicans (83%) said the president was doing an excellent or good job, including 47% who said he was doing an excellent job. A nearly identical share of Democrats (81%) rated his response as only fair or poor, including 56% who said it was poor.",
            "facts": [
                {
                    "fact_type": "proportion",
                    "fact_content": "83% of Republicans said the president was doing an excellent or good job."
                },
                {
                    "fact_type": "proportion",
                    "fact_content": "47% of Republicans said he was doing an excellent job."
                },
                {
                    "fact_type": "proportion",
                    "fact_content": "81% of Democrats rated his response as only fair or poor."
                },
                {
                    "fact_type": "proportion",
                    "fact_content": "56% of Democrats said it was poor."
                }
            ],
            "vis_data": [
                {
                    "x": "Republicans (Excellent/Good)",
                    "y": "83%",
                    "color": "#ff4d4d"
                },
                {
                    "x": "Republicans (Excellent)",
                    "y": "47%",
                    "color": "#ff5050"
                },
                {
                    "x": "Democrats (Fair/Poor)",
                    "y": "81%",
                    "color": "#009933"
                },
                {
                    "x": "Democrats (Poor)",
                    "y": "56%",
                    "color": "#53c653"
                }
            ],
            "titles": {
                "chart_title": "Partisan Views on Trump's COVID-19 Response (March 2020)",
                "x_axis": "Political Affiliation",
                "y_axis": "Percentage"
            },
            "article_meta_data": {
                "title": "A Year of U.S. Public Opinion on the Coronavirus Pandemic",
                "date": "March 5, 2021"
            },
            "vis_recommendation": "bar"
        }
    """

    user_prompt = f"""
    ####
    {paragraphs}
    ####
    """
    console.print("[bold yellow]Recommending Visualization...[/bold yellow]")
    story = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=ArticleVisRecommendation
    )
    console.print("[bold green]Recommending Visualization Completed![/bold green]")
    return json.loads(story)


@log_execution_time
def vis_criticizer(paragraphs):
    system_prompt = """
        You are an excellent assistant who provides detailed feedback on recommended visualization types for given data described in paragraphs.
        
        **Task:**
        Each paragraph comes with a recommended visualization type. Your task is to evaluate the suitability of the recommended visualization type for the data described in the paragraph.

        **Guidelines:**
        - Analyze whether the recommended visualization type fits the data values described.
        - Evaluate if the visualization type makes sense for the data.
        - Check the data values are structured correctly for the recommended visualization type. ex: data values with time should arrange increasing order in the vis_data array. 
        - If the recommended visualization is not suitable, suggest the most appropriate visualization type.
        - If the recommended visualization is suitable, but some of the data values are not suitable for the visualization type, suggest the data value removal or changes.
        - Provide clear feedback on why the recommended visualization type is not suitable.
       

        **Let's think step by step:**
        1. **Understand the Data:** Read the paragraph carefully to comprehend the data being described.
        2. **Assess the Recommendation:** Determine if the recommended visualization type appropriately represents the data.
        3. **Identify Issues:** If there's a mismatch, pinpoint why the recommended visualization is unsuitable.
        4. **Validate Data Structure:** Ensure the data values are structured and ordered correctly within the vis_data array for the recommended visualization type.
        4. **Suggest Alternatives:** Recommend a more suitable visualization type.
        5. **Explain Reasoning:** Provide a clear explanation for your suggestion.
        6. Check the folloiwing check list carefully. 
        
        **Check list** 
        - If the chart type is pie chart, make sure the sum of all the data points is 100%. If not, provide an unknown percentage for the missing part.
        - If the chart type is a bar chart and it only has one data point, suggest changing the chart type to isotype (if it is percentage or ratio) or text or pie chart.
        - If the chart is line chart, then all the y axis values should be numerical and same unit. 
        - Check all the data points are in same unit. If not, convert them to the same unit. (ex: if there is 1 million and 1 billion, convert them to the same unit as 1 million and 1000 million)
        - If the data values iclude times, then order them in ascending order. 
        
        
        Now, please apply this approach to the paragraphs provided between the #### delimiters.

        """

    user_prompt = f"""
        ####
        {paragraphs}
        ####
        """
    console.print("[bold yellow]Criticizing Recommendation...[/bold yellow]")
    feedback = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=ArticleVisRecommendationFeedback
    )
    console.print("[bold green]Criticizing Recommendation Completed![/bold green]")
    return json.loads(feedback)


@log_execution_time
def vis_refiner(paragraphs):
    system_prompt = """
        You are an excellent assistant for making correct decisions regarding visualization recommendations.
        **Task:**
        Your task is to evaluate the given visualization recommendation based on the provided feedback and adjust the visualization type if necessary.

        **Guidelines:**
        - Analyze the feedback related to the recommended visualization type.
        - Determine if the feedback is accurate and valid.
        - If the feedback is correct, suggest an alternative visualization type that better suits the data or requirements.
        - Maintain clarity and conciseness in your responses.

        **Let's think step by step:**
        1. Review the feedback provided for the recommended visualization type.
        2. Assess whether the feedback correctly identifies issues or improvements.
        3. If the feedback is valid, propose a more suitable visualization type.
        4. Justify your recommendation based on the analysis.
        5. Check the following checklist to refine the visualization recommendation:
        
        **Check list** 
        - If the chart type is pie chart, make sure the sum of all the data points is 100%. If not, provide an unknown percentage for the missing part.
        - If the chart type is a bar chart and it only has one data point, suggest changing the chart type to isotype (if it is percentage or ratio) or text or pie chart.
        - If the chart is line chart, then all the y axis values should be numerical and same unit. 
        - Check all the data points are in same unit. If not, convert them to the same unit. (ex: if there is 1 million and 1 billion, convert them to the same unit as 1 million and 1000 million)
        - If the data values iclude times, then order them in ascending order. 


    """

    user_prompt = f"""
    ####
    {paragraphs}
    ####
    """
    console.print("[bold yellow]Refining Recommendation...[/bold yellow]")
    vis_refined = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=ArticleVisRecommendation
    )
    console.print("[bold green]Refining Recommendation Completed![/bold green]")
    return json.loads(vis_refined)


@log_execution_time
def create_narrtive(paragraphs):
    system_prompt = """
        You are an excellent assistant in generating a concise narrative for a data story which is in html format.
        
        **Task:**
        Given a list of paragraphs and related data facts between #### delimeters, your task is to create a narrative that effectively conveys the key insight for the given facts in the paragraph. Please refer to the example to understand the html structure in the narrative provided between **** delimeters. 

        **Guidelines:**
        - The narrative should be less than 200 characters.
        - Format the narrative as HTML code using Tailwind CSS classes where applicable.
        - Color data values with their corresponding exact color codes as specified in the data using inline styles.
        - Color the important sentence parts with the specified hexadecimal color codes.
        - Ensure the narrative is clear, engaging, and accurately reflects the data provided.
        - Refer to the provided example for guidance.

        **Let's think step by step:**
        1. Take one paragraph at a time and read the data facts carefully.
        2. Identify the important data points that need to be highlighted.
        3. Assign the specified hexadecimal color codes to the corresponding data values in the narrative.
        4. Structure the narrative in HTML, using Tailwind CSS for general styling and inline styles for color-specific highlights.
        5. Review the narrative to ensure it is concise and within the character limit.
        6. Repeat the process for each paragraph to create a set of engaging narratives.

        **Example:**
        ****
            "narrative": "<p class='text-gray-800'>This is an example <span class='font-bold' style='color: #dc3545;'>Narrative</span>, which highlight numbers (<span class='font-bold' style='color: #28a745;'>20%%</span>), dates  <span class='font-bold' style='color: #28a745;'>August 2021 </span>, words <span class='font-bold' style='color: #28a745;'>increasing</span></p>"
        ****
    """

    user_prompt = f"""
        #### 
        {paragraphs}
        ####

    """

    console.print("[bold yellow]Creating Narrative...[/bold yellow]")
    vis_refined = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=ArticleVisRecommendation
    )
    console.print("[bold green]Creating Narrative Completed![/bold green]")
    return json.loads(vis_refined)


@log_execution_time
def organize_cluster_story(cluster_wise_facts):
    system_prompt = PROMPTS["ORGANIZE_CLUSTER_FACTS"]

    user_prompt = f"""
        #### 
        {cluster_wise_facts}
        ####

    """

    console.print("[bold yellow]Organizing Facts...[/bold yellow]")
    vis_refined = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=ClusterNarrative
    )
    console.print("[bold green]Organizing Facts Completed![/bold green]")
    return json.loads(vis_refined)


@log_execution_time
def cluster_topics(paragraphs):
    system_prompt = """
       You are a skilled assistant. 
       
       Your task is to provide cluster the given paragrpahs (between ****) into topics.
       Each paragraph can be assigned to one or more topics.
       Make sure the given order_id is remain same in the output.
       Make sure number of topics is less than 10.
    """

    user_prompt = f"""
    
    **** 
    {paragraphs}
    **** 
    
    """
    console.print("[bold yellow]Clustering...[/bold yellow]")
    topics = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=Clusters
    )
    console.print("[bold green]Clustering completed![/bold green]")

    return json.loads(topics)


@log_execution_time
def cluster_detail_generation(cluster, search_query):
    system_prompt = PROMPTS["CLUSTER_DETAIL_GENERATION"]

    user_prompt = f"""
    
    ####
    {cluster}
    ####
    
    **** 
    {search_query}
    **** 
    """
    console.print("[bold yellow]Creating Detail Cluster...[/bold yellow]")
    topics = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=DetailCluster
    )
    console.print("[bold green]Creating Detail Cluster Completed![/bold green]")

    return json.loads(topics)


@log_execution_time
def refine_cluster_detail(cluster, search_query):
    system_prompt = PROMPTS["REFINE_CLUSTER_DETAIL"]

    user_prompt = f"""
    
    ####
    {cluster}
    ####
    
    ****
    {search_query}
    ****
    
    """
    console.print("[bold yellow]Refining Detail Cluster...[/bold yellow]")
    topics = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=DetailClusters
    )
    console.print("[bold green]Refining Detail Cluster Completed![/bold green]")

    return json.loads(topics)


@log_execution_time
def clickbait_generation(cluster, search_query):
    system_prompt = PROMPTS["CLICKBAIT_GENERATION"]

    user_prompt = f"""
    
    #### 
    {cluster}
    #### 
    
    ****
    {search_query}
    ****
    
    """
    console.print("[bold yellow]Creating Clickbait...[/bold yellow]")
    topics = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=ClusterClickbait
    )
    console.print("[bold green]Creating Clickbait Completed![/bold green]")

    return json.loads(topics)


def organize_data_by_topic(input_data):
    # Initialize a dictionary to store topics and corresponding data
    topics_dict = defaultdict(list)

    # Process the input data
    for item in input_data:
        for topic in item["topics"]:
            topic_name = topic["topic"]
            cluster_id = topic["cluster_id"]

            # Collect the data related to the topic
            topic_data = {
                "paragraph": item["paragraph"],
                "facts": item["facts"],  # Ensure this is a list
                "vis_data": item["vis_data"],
                "titles": item["titles"],
                "article_meta_data": item["article_meta_data"],
                "relatedness_score": item["relatedness_score"],
                "order_id": item["order_id"],
                "narrative": item["narrative"],
                "vis_recommendation": item["vis_recommendation"],
            }

            # Add the data to the corresponding topic
            topics_dict[topic_name].append(
                {"cluster_id": cluster_id, "data": [topic_data]}
            )

    # Convert topics_dict into the desired output format
    output_data = [
        {
            "topic": topic_name,
            "cluster_id": topics_data[0]["cluster_id"],
            "data": [item["data"][0] for item in topics_data],
        }
        for topic_name, topics_data in topics_dict.items()
    ]

    return output_data


def remove_same_facts(cluster):
    fact_groups = cluster["fact_groups"]
    cluster_id = cluster["cluster_id"]
    return [
        {
            "fact_group_id": f"{cluster_id}_{index}",
            "fact_group_content": group[0]["fact_content"],
            "fact_count": len(group),
            "article_count": len(set(fact["fact_id"].split("_")[0] for fact in group)),
            "fact_ids": [fact["fact_id"] for fact in group],
            "article_ids": list(set(fact["fact_id"].split("_")[0] for fact in group)),
        }
        for index, group in enumerate(fact_groups)
    ]


def structure_fact_groups(fact_groups, results):
    fact_dict = {fact["fact_id"]: fact for fact in results["all_facts_with_vis_data"]}

    all_fact_gropus = []
    cluster_wise_facts = []

    for cluster in fact_groups:
        new_facts = []
        for item in cluster["fact_groups"]:
            new_facts.append(
                {
                    "fact_group_id": item["fact_group_id"],
                    "fact_group_content": item["fact_group_content"],
                }
            )
            all_fact_gropus.append(item)
            facts = []
            for fact_id in item["fact_ids"]:
                facts.append(fact_dict[fact_id])
            item["facts"] = facts
        new_cluster = {
            "cluster_id": cluster["cluster_id"],
            "cluster_size": len(new_facts),
            "facts": new_facts,
        }
        cluster_wise_facts.append(new_cluster)

    return {
        "clusters": fact_groups,
        "cluster_wise_facts": cluster_wise_facts,
        "all_fact_groups": all_fact_gropus,
    }


def identify_similar_facts(facts):
    system_prompt = PROMPTS["IDENTIFY_SIMILAR_FACTS"]

    user_prompt = f"""
    ####
    {facts}
    ####
    """

    console.print("[bold yellow]Identifying Similar Facts...[/bold yellow]")
    similar_facts = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=FactGroups
    )
    console.print("[bold green]Identifying Similar Facts Completed![/bold green]")

    if similar_facts is None:
        return None

    groups = json.loads(similar_facts)
    cleaned_groups = remove_same_facts(groups)
    groups["fact_groups"] = cleaned_groups
    return groups


def assign_id(merged_facts):
    cluster_id = merged_facts["cluster_id"]
    facts = merged_facts["merged_facts"]
    for index, fact in enumerate(facts):
        fact["merged_id"] = f"{cluster_id}_{index}"
    return merged_facts


def merge_facts(facts):
    system_prompt = PROMPTS["MERGE_FACTS"]

    user_prompt = f"""
    ####
    {facts}
    ####
    """

    console.print("[bold yellow]Merging Facts...[/bold yellow]")
    merged_facts = gpt_helper.ask_gpt_with_response_format(
        system_prompt,
        user_prompt,
        response_format=MergedFacts,
        temperature=0.2,
        top_p=0.1,
    )
    merged_facts = assign_id(json.loads(merged_facts))

    console.print("[bold green]Merging Facts Completed![/bold green]")
    return merged_facts


@log_execution_time
def validate_merged_facts(facts):
    system_prompt = PROMPTS["VALIDATE_MERGED_FACTS"]

    user_prompt = f"""
    ####
    {facts}
    ####
    """

    console.print("[bold yellow]Checking Merged Facts...[/bold yellow]")
    errors = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=Errors
    )
    console.print("[bold green]Checking Merged Facts Completed![/bold green]")
    return json.loads(errors)


@log_execution_time
def correct_merged_facts(facts):
    system_prompt = PROMPTS["CORRECT_MERGED_FACTS"]

    user_prompt = f"""
    ####
    {facts}
    ####
    """

    console.print("[bold yellow]Correcting merged facts...[/bold yellow]")
    merged_facts = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=MergedFacts
    )
    merged_facts = assign_id(json.loads(merged_facts))
    console.print("[bold green]Correcting merged facts Completed![/bold green]")
    return merged_facts


@log_execution_time
def refine_merged_facts(facts):
    system_prompt = PROMPTS["REFINE_MERGED_FACTS"]

    user_prompt = f"""
    ####
    {facts}
    ####
    """

    console.print("[bold yellow]Correcting Merged Facts...[/bold yellow]")
    merged_facts = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=MergedFacts
    )
    merged_facts = assign_id(json.loads(merged_facts))
    console.print("[bold green]Correcting Merged Facts Completed![/bold green]")
    return merged_facts


@log_execution_time
def get_entities_in_merged_facts(merged_fact):
    system_prompt = PROMPTS["IDENTIFY_ENTITIES_IN__MERGED_FACTS"]

    user_prompt = f"""
    ####
    {merged_fact}
    ####
    """

    console.print("[bold yellow]Identify Subjects In Merged Facts...[/bold yellow]")
    fact_entities = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=MergedFactEntities
    )
    console.print(
        "[bold green]Identify Subjects In Merged Facts Completed![/bold green]"
    )
    return json.loads(fact_entities)


@log_execution_time
def fill_missing_entities(fact, article):
    system_prompt = PROMPTS["FILL_MISSING_ENTITIES"]

    user_prompt = f"""
    ####
    {fact}
    ####
    
    ****
    {article}
    **** 
    """

    console.print("[bold yellow]Fill Missing Entities...[/bold yellow]")
    fact_entities = gpt_helper.ask_gpt_with_response_format(
        system_prompt, user_prompt, response_format=FactGroupWithMissingEntity
    )
    console.print("[bold green]Fill Missing Entities Completed![/bold green]")
    return json.loads(fact_entities)


@log_execution_time
def handle_filling_data(cluster_missing_entities, articles):
    def process_fact_group(fact_group):
        article_id = fact_group["article_ids"][0]
        return fill_missing_entities(fact_group, articles[int(article_id)])

    def process_merged_fact(merged_fact):
        with ThreadPoolExecutor() as executor:
            merged_fact["facts"] = list(
                executor.map(process_fact_group, merged_fact["facts"])
            )
        return merged_fact

    def process_cluster(cluster):
        with ThreadPoolExecutor() as executor:
            cluster["merged_facts"] = list(
                executor.map(process_merged_fact, cluster["merged_facts"])
            )
        return cluster

    with ThreadPoolExecutor() as executor:
        cluster_missing_entities = list(
            executor.map(process_cluster, cluster_missing_entities)
        )

    return cluster_missing_entities
