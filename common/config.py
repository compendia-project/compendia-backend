from prompts.prompt_loader import load_prompt_from_file

MODEL_CONFIG = {
    "DEFAULT_MODEL": "gpt-4o-2024-08-06",
    "DEFAULT_TEMPERATURE": 0.7,
    "FALLBACK_MODEL": "gpt-4o-mini",
    "EMBEDDING_MODEL": "text-embedding-3-large",
    "MAX_INPUT_TOKENS": 128000,
    "MAX_OUTPUT_TOKENS": 16400,
    "TOKEN_SAFETY_MARGIN": 10000,
    "TOP_P": 0.1,
}

TEMPLATE_CONFIGS = {"TEMPLATE_PATH": "templates/FactSheetD3.html"}

CLUSTER_CONFIGS = {"MAX_CLUSTER_SIZE": 11, "FACTS_PER_CLUSTER": 8}

DATE_CONFIGS = {"DATE_FORMAT": "%Y-%m-%d"}

THEME_CONFIGS = {"COLORS": ["#22d3ee", "#2dd4bf", "#f87171", "#facc15"]}

PROMPTS = {
    "GENERATE_SEARCH_QUERIES": load_prompt_from_file(
        "prompts/1_generate_search_queries.txt"
    ),
    "EXTRACT_FILTER_PARA": load_prompt_from_file(
        "prompts/2_extract_and_filter_para.txt"
    ),
    "EXTRACT_FACTS": load_prompt_from_file("prompts/3_extract_facts.txt"),
    "DATA_EXTRACTION": load_prompt_from_file("prompts/4_data_extraction.txt"),
    "VALIDATE_DATA_EXTRACTION": load_prompt_from_file(
        "prompts/5_validate_data_extraction.txt"
    ),
    "REFINE_DATA_EXTRACTION": load_prompt_from_file(
        "prompts/6_refine_data_extraction.txt"
    ),
    "IDENTIFY_SIMILAR_FACTS": load_prompt_from_file(
        "prompts/7_identify_similar_facts.txt"
    ),
    "MERGE_FACTS": load_prompt_from_file("prompts/8_merge_facts.txt"),
    "FILL_MISSING_ENTITIES": load_prompt_from_file(
        "prompts/9_fill_missing_entities.txt"
    ),
    "REFINE_MERGED_FACTS": load_prompt_from_file("prompts/10_refine_merged_facts.txt"),
    "VALIDATE_MERGED_FACTS": load_prompt_from_file(
        "prompts/11_validate_merged_facts.txt"
    ),
    "CORRECT_MERGED_FACTS": load_prompt_from_file(
        "prompts/12_correct_merged_facts.txt"
    ),
    "CLICKBAIT_GENERATION": load_prompt_from_file(
        "prompts/13_clickbait_generation.txt"
    ),
    "CLUSTER_DETAIL_GENERATION": load_prompt_from_file(
        "prompts/13_cluster_detail_generation.txt"
    ),
    "REFINE_CLUSTER_DETAIL": load_prompt_from_file(
        "prompts/14_refine_cluster_detail.txt"
    ),
    "ORGANIZE_CLUSTER_FACTS": load_prompt_from_file(
        "prompts/14_organize_cluster_facts.txt"
    ),
    "STYLE_NARRATIVE": load_prompt_from_file("prompts/15_style_narrative.txt"),
    "REFINE_NARRATIVE": load_prompt_from_file("prompts/16_refine_narrative.txt"),
    "IDENTIFY_ENTITIES_IN__MERGED_FACTS": load_prompt_from_file(
        "prompts/identify_entities_in_merged_facts.txt"
    ),
}
