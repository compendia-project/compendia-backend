from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, confloat


class FactType(Enum):
    """Enum for different types of data facts."""

    VALUE = "value"
    DIFFERENCE = "difference"
    PROPORTION = "proportion"
    TREND = "trend"
    # CATEGORIZATION = "categorization"
    DISTRIBUTION = "distribution"
    RANK = "rank"
    # ASSOCIATION = "association"
    EXTREME = "extreme"
    # OUTLIER = "outlier"


class VisualisationType(Enum):
    """Enum for different types of visualisations."""

    BAR = "bar"
    LINE = "line"
    ISO_TYPE = "isotype"
    MAP = "map"
    # SCATTER_PLOT = "scatter_plot"
    PIE = "pie"
    # AREA = "area"
    # BUBBLE = "bubble"
    TEXT = "text"
    RANGE = "range"
    # TABLE = "table"
    # BOX_PLOT = "box_plot"
    # TREEMAP = "treemap"


class ArticleMetaData(BaseModel):
    """Data model for metadata of an article."""

    title: str = Field(description="The title of the article.")
    date: str = Field(description="The date the article was published.")
    # source: str = Field(description="The source of the article.")
    url: str = Field(description="The URL of the article.")


class Article(BaseModel):
    """Article or document details"""

    # title: str = Field(description="title of the article or document")
    paragraphs: List[str] = Field(description="extracted paragraphs from the article")


class Article_v2(BaseModel):
    """Article or document details"""

    title: str = Field(description="title of the article or document")
    date: str = Field(description="date of the article or document")
    paragraphs: List[str] = Field(description="extracted paragraphs from the article")


class ParagraphWithScore(BaseModel):
    """Data model for a paragraph with a relatedness score."""

    paragraph: str = Field(description="The paragraph of text.")
    score: float = Field(
        ge=0, le=1, description="The relatedness score of the paragraph."
    )


class DataFact(BaseModel):
    """Data model for a data fact."""

    fact_type: FactType = Field(
        description="The type of data fact extracted from the paragraph."
    )
    fact_content: str = Field(description="The data fact extracted from the paragraph.")


class DataFactWithParagraph(BaseModel):
    """Data model for a data fact with the paragraph it was extracted from."""

    paragraph: str = Field(
        description="The paragraph from which the data fact was extracted."
    )
    # fact_type: FactType = Field(
    #     description="The type of data fact extracted from the paragraph."
    # )
    facts: List[DataFact] = Field(
        description="The data facts extracted from the paragraph."
    )


class DataFactWithRelatedSentence(DataFactWithParagraph):
    """Extended data model that includes a related sentence."""

    related_sentence: str = Field(
        description="A related sentence to the data fact from the paragraph."
    )


class DataValue(BaseModel):
    """Data model for a data value."""

    label: str = Field(description="The label/category of the data.")
    value: str = Field(description="The value of the data.")
    unit: str = Field(description="The scale unit of the data.")
    # color: str = Field(description="The color value of the data.")
    # unit: str = Field(description="The scale unit of the data.")
    # label: str = Field(description="The label of the data.")


class DataValueColor(DataValue):
    color: str = Field(description="The color value of the data.")


class ChartTitle(BaseModel):
    chart_title: str = Field(description="The title of the chart.")
    x_axis: str = Field(description="The x-axis of the chart.")
    y_axis: str = Field(description="The y-axis of the chart.")


class FactWithVisRecommendation(DataFact):
    """Data model for a data fact with a visualization recommendation."""

    # vis_recommendation: VisualisationType = Field(
    #     description="The visualization recommendation for the data fact."
    # )
    vis_data: List[DataValue] = Field(
        description="list of data values for visualization"
    )
    # titles: ChartTitle = Field(
    #     description="chart title, x axis title, and y axis title"
    # )


class FactVisId(FactWithVisRecommendation):
    """Data model for a data fact with a visualization recommendation."""

    fact_id: str = Field(description="The id of the fact.")


class FactWithVisRecommendationError(FactWithVisRecommendation):
    """Data model for a data fact with a visualization recommendation error."""

    error: List[str] = Field(
        description="The error in the visualization recommendation."
    )


class DataFactWithVis(BaseModel):
    """Data model for a data fact with the paragraph it was extracted from."""

    paragraph: str = Field(
        description="The paragraph from which the data fact was extracted."
    )
    facts: List[FactWithVisRecommendation] = Field(
        description="The data facts with visualization recommendation"
    )


class DataFactWithVisError(BaseModel):
    """Data model for a data fact with a visualization recommendation error."""

    paragraph: str = Field(
        description="The paragraph from which the data fact was extracted."
    )
    facts: List[FactWithVisRecommendationError] = Field(
        description="The data facts with visualization recommendation error"
    )


class DataFactWithVisData(DataFactWithParagraph):
    """Extended data model that includes a visualization data."""

    vis_data: List[DataValue] = Field(
        description="list of data values for visualization"
    )
    titles: ChartTitle = Field(
        description="chart title, x axis title, and y axis title"
    )


class DataFactWithMetaData(DataFactWithVisData):
    """Extended data model that includes meta data."""

    article_meta_data: ArticleMetaData = Field(description="meta data of the article")


class DataFactWithRelatedness(DataFactWithMetaData):
    """Extended data model that includes relatedness score."""

    relatedness_score: float = Field(description="relatedness score of the paragraph")


class OrderedDataFactWithMetaData(DataFactWithRelatedness):
    """Extended data model that includes order of the data fact."""

    order_id: int = Field(description="order of the data fact in the article")
    narrative: str = Field(description="narrative of the data fact")


class VisRecomendation(OrderedDataFactWithMetaData):
    """Extended data model that includes visualization recommendation."""

    vis_recommendation: VisualisationType = Field(
        description="list of recommended visualizations"
    )


class VisRecomendationFeedback(VisRecomendation):
    """Extended data model that includes visualization recommendation feedback."""

    recommendation_feedback: str = Field(
        description="feebdack on the visualization recommendation"
    )


class DataFactWithVisDataError(DataFactWithVisData):
    """Extended data model that includes extracted data error."""

    Errors: List[str] = Field(description="errors identified in the extracted data")


class ValidationOutput(BaseModel):
    """Data model for validation output."""

    has_error: bool = Field(
        description="True if there is an error in the extracted data, False otherwise."
    )
    vis_data_error: List[DataFactWithVisError] = Field(
        description="list of data values for visualization with error"
    )


class ArticleDataFacts(BaseModel):
    """Data model for data facts extracted from an article."""

    # title: str = Field(description="The title of the article.")
    data_facts_with_para: List[DataFactWithParagraph] = Field(
        description="The data facts with paragraphs extracted from the article."
    )


class ArticleDataFactSentence(BaseModel):
    """Data model for data facts extracted from an article."""

    # title: str = Field(description="The title of the article.")
    data_facts_with_sentence: List[DataFactWithRelatedSentence] = Field(
        description="The data facts with paragraphs and related sentence extracted from the article."
    )


class ArticleDataFactVisData(BaseModel):
    """Data model for data facts extracted from an article."""

    # title: str = Field(description="The title of the article.")
    data_facts_with_vis_data: List[DataFactWithVis] = Field(
        description="The data facts with paragraphs and data values extracted from the article."
    )


class ArticleDataFactVisDataMeta(BaseModel):
    """Data model for data facts extracted from an article."""

    data_facts_with_vis_data_meta: List[DataFactWithMetaData] = Field(
        description="The data fact with paragraphs, meta data and data values extracted from the article"
    )


class ArticleDataFactVisDataMetaRelatedness(BaseModel):
    """Data model for data facts extracted from an article."""

    data_facts_with_vis_data_meta_relatedness: List[DataFactWithRelatedness] = Field(
        description="The data fact with paragraphs, meta data, data values and relatedness score extracted from the article"
    )


class ArticleDataFactVisDataMetaOrder(BaseModel):
    """Data model for data facts extracted from an article."""

    data_facts_with_vis_data_meta_order: List[OrderedDataFactWithMetaData] = Field(
        description="The data fact with paragraphs, meta data, data values and order extracted from the article"
    )


class ArticleVisRecommendation(BaseModel):
    """Data model for data facts extracted from an article."""

    data_facts_with_vis_recommendation: List[VisRecomendation] = Field(
        description="The data fact with paragraphs, meta data, data values, order, narrative and visualization recommendation extracted from the article"
    )


class ArticleVisRecommendationFeedback(BaseModel):
    """Data model for data facts extracted from an article."""

    vis_recommendation_feedbacks: List[VisRecomendationFeedback] = Field(
        description="The data fact with paragraphs, meta data, data values, order, narrative, visualization recommendation and feedback extracted from the article"
    )


class ChartValidator(BaseModel):
    """Html code component validation results"""

    is_valid: bool = Field(
        description="True if the chart components are valid, False otherwise."
    )
    issues: List[str] = Field(
        description="List of issues identified in the chart components."
    )


class DataStoryPiece(BaseModel):
    """Data model for a piece of a data story."""

    order_id: int = Field(description="The order in which the story piece appears.")
    fact_type: FactType = Field(
        description="The type of fact the story piece is presenting."
    )
    paragraph: str = Field(
        description="The paragraph of text the piece of the data story."
    )
    related_sentence: str = Field(
        description="A sentence closely related to the data presented in the data story piece."
    )
    summary: str = Field(..., description="A summary of the data story piece.")
    visualisation_types: List[VisualisationType] = Field(
        description="The types of visualisations used in the data story piece."
    )
    # confidence: float = Field(ge=0, le=1, description="The confidence score of the data story piece.")


class Story(BaseModel):
    """Data model for a data story."""

    stories: List[DataStoryPiece] = Field(
        ...,
        description="A list of data story pieces that make up the entire data story.",
    )


class Overview(BaseModel):
    """ "Overview of the facts"""

    summary: str = Field(description="Summary of the story")


class GroupedFacts(BaseModel):
    """Grouped facts"""

    topic: str = Field(description="topic of the group")
    topic_color: str = Field(description="color of the topic")
    facts: List[VisRecomendation] = Field(description="Facts in the group")


class StoryLine(BaseModel):
    """Story line of the facts"""

    start_point: str = Field(description="Starting point of the story")
    grouped_facts: List[GroupedFacts] = Field(description="Grouped facts")


class SearchQueryList(BaseModel):
    """Data model for a list of search queries."""

    search_queries: List[str] = Field(description="A list of search queries.")


class Topic(BaseModel):
    """Data model for a topic."""

    topic: str = Field(description="The topic of the cluster.")
    cluster_id: str = Field(description="The ID of the cluster.")
    description: str = Field(description="The description of the cluster.")


class Cluster(BaseModel):
    """Data model for a cluster of articles."""

    paragraph: str = Field(description="The paragraph of text.")
    order_id: int = Field(description="The order of the paragraph.")
    topics: List[Topic] = Field(description="The topics of the cluster.")


class Clusters(BaseModel):
    """Data model for a cluster of articles."""

    clusters: List[Cluster] = Field(description="A list of clusters.")


class Fact(BaseModel):
    """Data model for a fact."""

    fact_content: str = Field(description="The fact content.")
    fact_id: str = Field(description="Id of the fact.")


class GroupedFact(BaseModel):
    """Data model for a group of facts."""

    fact_group_id: str = Field(description="Id of the fact group.")
    fact_group_content: str = Field(description="Fact content.")


class MergedFact_(BaseModel):
    """Data model for merged facts."""

    merged_id: str = Field(description="Id of the merged fact.")
    merged_content: str = Field(description="The merged fact content.")


class Clickbait(BaseModel):
    """Data model for a clickbait."""

    clickbait: str = Field(description="The clickbait.")
    related_facts: List[GroupedFact] = Field(
        description="The facts related to the clickbait."
    )
    number_of_facts: int = Field(
        description="The number of facts related to the clickbait."
    )


class DetailCluster(BaseModel):
    """Data model for a detailed cluster."""

    cluster_id: int = Field(description="The id of the cluster.")
    title: str = Field(description="The title of the cluster.")
    description: str = Field(description="The full summary description of the cluster.")
    representative_facts: List[Fact] = Field(
        description="The representative facts of the cluster."
    )


class OrderedDetailCluster(DetailCluster):
    cluster_order_id: int = Field(description="The order of the cluster in the story.")


class DetailClusters(BaseModel):
    """Data model for a detailed clusters."""

    clusters: List[OrderedDetailCluster] = Field(
        description="The list of detailed clusters."
    )


class ClusterClickbait(BaseModel):
    """Data model for a list of clickbait."""

    cluster_id: int = Field(description="The id of the cluster.")
    clickbait_list: List[Clickbait] = Field(description="The list of clickbait.")
    important_words: List[str] = Field(
        description="The list of important words from all facts"
    )


class FactNarrative(MergedFact_):
    """Data model for a fact narrative."""

    order_id: int = Field(description="The order of the fact.")
    narrative: str = Field(description="The narrative of the fact.")


class ClusterNarrative(BaseModel):
    """Data model for a cluster narrative."""

    cluster_id: int = Field(description="The id of the cluster.")
    number_of_facts: int = Field(description="The number of facts in the cluster.")
    facts: List[FactNarrative] = Field(description="The facts in the cluster.")


class StyledNarrative(BaseModel):
    """Data model for a styled narrative."""

    narrative: str = Field(description="The narrative of the fact in html format.")
    vis_data: List[DataValueColor] = Field(
        description="The data values for visualization."
    )
    vis_recommendation: VisualisationType = Field(
        description="The visualization recommendation for the data fact."
    )


class FactGroup(BaseModel):
    """Data model for a fact group."""

    fact_group: List[Fact] = Field(description="The list of facts in the group.")


class ClusterFactGroup(BaseModel):
    """Data model for a fact group."""

    cluster_id: str = Field(description="The id of the cluster.")
    fact_group: List[FactGroup] = Field(description="The list of facts in the group.")


class FactGroups(BaseModel):
    """Data model for a list of fact groups."""

    cluster_id: str = Field(description="The id of the cluster.")
    fact_groups: List[List[Fact]] = Field(description="The list of fact groups.")


class FactGroupWithArticleData(BaseModel):
    """Data model for a fact group with article data."""

    fact_group_id: str = Field(description="The id of the fact group.")
    fact_group_content: str = Field(description="Fact content.")
    fact_ids: List[str] = Field(description="IDs of the facts in the group.")
    article_ids: List[str] = Field(description="IDs of the articles in the group.")
    facts: List[FactVisId] = Field(description="The list of facts in the group.")


class FactGroupWithMissingEntity(FactGroupWithArticleData):
    """Data model for a fact group with missing entities."""

    missing_entities: List[str] = Field(description="The missing entities in the fact.")


class MergedFact(BaseModel):
    """Data model for merged facts."""

    merged_content: str = Field(description="The merged fact content.")
    merged_data: List[DataValue] = Field(description="The merged data values.")
    merged_recommendation: VisualisationType = Field(
        description="The merged visualization recommendation."
    )
    titles: ChartTitle = Field(
        description="The chart title, x axis title, and y axis title."
    )
    facts: List[FactGroupWithArticleData] = Field(
        description="The list of merged facts."
    )


class Entity(BaseModel):
    """Represents an extracted entity (e.g., person, location, date)."""

    text: str
    type: str  # e.g., "GPE", "DATE", "PERCENT"
    # normalized_value: Optional[str] = None  # e.g., ISO date, standardized name
    # confidence: Optional[str] = None  # 0-1 confidence score
    # attributes: Optional[Dict] = {}  # domain-specific metadata (e.g., sentiment)


class EventArgument(BaseModel):
    """An argument participating in an event (e.g., subject, time)."""

    role: str = Field(
        description="The role of the argument in the event (e.g., 'subject', 'time', 'rate')"
    )
    text: str = Field(description="The text of the argument")
    entity_type: str = Field(description="The linked entity type (e.g., 'PERCENT')")
    normalized_value: str = Field(description="The normalized value of the argument")


class Event(BaseModel):
    """Represents an extracted event (e.g., economic growth, merger)."""

    trigger: str = Field(
        description="The word/phrase that triggered the event (e.g., 'grown')"
    )
    type: str = Field(
        description="The type of the event (e.g., 'EconomicGrowth', 'Marriage')"
    )
    arguments: List[EventArgument] = Field(
        description="The list of arguments participating in the event"
    )
    time: str = Field(description="The time of the event (e.g., '2021-2022')")
    place: str = Field(description="The place of the event (e.g., 'UK')")


class MergedFactEntities(MergedFact):
    """Data model for merged facts."""

    entities: List[List[Entity]] = Field(description="The list of entities per fact.")
    events: List[Event] = Field(description="The list of events per fact.")


class MergedFacts(BaseModel):
    """Data model for merged facts."""

    cluster_id: str = Field(description="The id of the cluster.")
    merged_facts: List[MergedFact] = Field(description="The list of merged facts.")


class MergedFactsEntities(BaseModel):

    cluster_id: str = Field(description="The id of the cluster.")
    merged_facts: List[MergedFactEntities] = Field(
        description="The list of merged facts."
    )


class Errors(BaseModel):
    """Data model for errors."""

    errors: List[str] = Field(description="List of errors.")


class FilledFact(BaseModel):
    """Data model for a filled fact."""

    fact_id: str = Field(description="The id of the fact.")
    fact_content: str = Field(description="The content of the fact.")
    vis_data: List[DataValue] = Field(description="The data values for visualization.")
    missing_entities: List[str] = Field(description="The missing entities in the fact.")
