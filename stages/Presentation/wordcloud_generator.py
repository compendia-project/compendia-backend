import re
from collections import defaultdict
from xml.dom import minidom  # For more reliable XML handling

import numpy as np
from scipy.interpolate import interp1d
from wordcloud import STOPWORDS, WordCloud


def get_radius(max_original_facts, value):
    radius_scale = interp1d([1, max_original_facts], [40, 60], fill_value="extrapolate")
    return int(radius_scale(value))


def get_frequencies(clickbait_list):
    word_frequencies = defaultdict(int)

    def update_frequencies(text, count, word_frequencies):
        words = re.findall(r"\b\w+\b", text)
        for word in words:
            word_frequencies[word] += count

    for item in clickbait_list:
        # word_frequencies[item["clickbait"]] = item["number_of_facts"]
        update_frequencies(item["clickbait"], item["number_of_facts"], word_frequencies)

    word_frequencies = dict(word_frequencies)
    return word_frequencies


def get_frequency_for_text(text):
    # Dictionary to store frequency of words and numbers
    frequency = defaultdict(int)

    # List of common stopwords
    stopwords = {
        "is",
        "by",
        "of",
        "the",
        "and",
        "a",
        "an",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "from",
        "up",
        "down",
        "out",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
        "don",
        "should",
        "now",
    }

    # Regular expression to extract words and numbers
    tokens = re.findall(r"\b\d+\.?\d*\b|\b[a-zA-Z]+\b", text)

    # Count frequency of each token
    for token in tokens:
        lower_token = token.lower()
        if lower_token not in stopwords:
            frequency[
                lower_token
            ] += 1  # Convert words to lowercase for case-insensitive matching

    return dict(frequency)


def get_most_talked_facts(cluster):
    text = ""
    max_original_facts_per_group = 0
    most_talked_facts = []

    for fact_group in cluster["all_fact_groups"]:
        if max_original_facts_per_group < fact_group["number_of_similar_facts"]:
            max_original_facts_per_group = fact_group["number_of_similar_facts"]

    for fact_group in cluster["all_fact_groups"]:
        if fact_group["number_of_similar_facts"] == max_original_facts_per_group:
            most_talked_facts.append(fact_group["facts"])

    for most_talked_fact in most_talked_facts:
        for fact in most_talked_fact:
            text += fact["fact_content"] + " "
    return text


def generate_outer_wordcloud(cluster, max_riginal_facts, cluster_wise_fact):
    text = ""
    for item in cluster_wise_fact["facts"]:
        text += item["fact_content"] + " "

    stopwords = set(STOPWORDS)

    number_of_original_facts = cluster["number_of_original_facts"]
    clickbait_list = cluster["clickbait_list"]

    # word_frequencies = get_frequencies(clickbait_list)

    word_frequencies = cluster["important_word_frequency"]

    # word_frequencies = get_frequency_for_text(text)

    # word_frequencies = get_frequencies(clickbait_list)
    # Create proportional mask parameters
    grid_size = 200
    center = (grid_size // 2, grid_size // 2)
    outer_radius = grid_size // 2 - 2
    inner_radius = get_radius(max_riginal_facts, number_of_original_facts) + 2

    # Create high-res mask
    x, y = np.ogrid[:grid_size, :grid_size]
    outer_mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 > outer_radius**2
    inner_mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 < inner_radius**2
    mask = outer_mask | inner_mask
    mask = 255 * mask.astype(np.uint8)

    wc = WordCloud(
        background_color=None,
        mode="RGBA",
        mask=mask,
        scale=1,
        max_font_size=20,
        min_font_size=10,
        width=grid_size,
        height=grid_size,
        random_state=42,
        max_words=50,
        # repeat=True,
        prefer_horizontal=1,
        stopwords=stopwords,
    )
    wc.generate_from_frequencies(word_frequencies)

    # wc.generate(text)

    svg_content = wc.to_svg()
    doc = minidom.parseString(svg_content)
    svg_element = doc.documentElement

    new_group = doc.createElementNS("http://www.w3.org/2000/svg", "g")
    new_group.setAttribute("transform", "translate(-100 -100)")

    children = list(svg_element.childNodes)
    for child in children:
        if child.nodeType == child.ELEMENT_NODE:
            svg_element.removeChild(child)
            child.setAttribute("style", "fill: #b8ccd6")
            new_group.appendChild(child)

    new_doc = minidom.Document()
    new_doc.appendChild(new_group.cloneNode(deep=True))
    group_content = new_doc.documentElement.toxml()
    return group_content


def generate_inner_wordcloud(cluster, max_riginal_facts, cluster_wise_fact):
    text = ""
    for item in cluster_wise_fact["facts"]:
        text += item["fact_content"] + " "

    stopwords = set(STOPWORDS)

    text = get_most_talked_facts(cluster)

    number_of_original_facts = cluster["number_of_original_facts"]
    clickbait_list = cluster["clickbait_list"]

    word_frequencies = get_frequency_for_text(text)

    inner_radius = get_radius(max_riginal_facts, number_of_original_facts)
    grid_size = inner_radius * 2
    center = (grid_size // 2, grid_size // 2)

    x, y = np.ogrid[:grid_size, :grid_size]
    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 > inner_radius**2
    mask = 255 * mask.astype(np.uint8)

    wc = WordCloud(
        background_color=None,
        mode="RGBA",
        mask=mask,
        scale=1,
        max_font_size=20,
        min_font_size=2,
        width=grid_size,
        height=grid_size,
        random_state=42,
        max_words=20,
        repeat=True,
        prefer_horizontal=1,
        stopwords=stopwords,
    )
    # wc.generate_from_frequencies(word_frequencies)
    wc.generate(text)

    svg_content = wc.to_svg()
    doc = minidom.parseString(svg_content)
    svg_element = doc.documentElement

    new_group = doc.createElementNS("http://www.w3.org/2000/svg", "g")
    new_group.setAttribute("transform", f"translate({-inner_radius} {-inner_radius})")

    children = list(svg_element.childNodes)
    for child in children:
        if child.nodeType == child.ELEMENT_NODE:
            svg_element.removeChild(child)
            child.setAttribute("style", "fill: white")
            new_group.appendChild(child)

    new_doc = minidom.Document()
    new_doc.appendChild(new_group.cloneNode(deep=True))
    group_content = new_doc.documentElement.toxml()
    return group_content


def generate_wordcloud(cluster, max_riginal_facts, cluster_wise_fact):

    svg_content_outer = generate_outer_wordcloud(
        cluster, max_riginal_facts, cluster_wise_fact
    )
    svg_content_inner = generate_inner_wordcloud(
        cluster, max_riginal_facts, cluster_wise_fact
    )

    cluster["word_cloud"] = {
        "outer": svg_content_outer,
        "inner": svg_content_inner,
    }

    return cluster
