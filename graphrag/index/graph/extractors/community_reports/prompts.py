# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""A file containing prompts definition."""

COMMUNITY_REPORT_PROMPT = """
You are an AI assistant that helps a human analyst to perform general information discovery. Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

# Goal
Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

# Report Structure

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
        ]
    }}

# Grounding Rules

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (1), Entities (5, 7); Relationships (23); Claims (7, 2, 34, 64, 46, +more)]."

where 1, 5, 7, 23, 2, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.


# Example Input
-----------
Text:

Entities

id,entity,description
5,VERDANT OASIS PLAZA,Verdant Oasis Plaza is the location of the Unity March
6,HARMONY ASSEMBLY,Harmony Assembly is an organization that is holding a march at Verdant Oasis Plaza

Relationships

id,source,target,description
37,VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza is the location of the Unity March
38,VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly is holding a march at Verdant Oasis Plaza
39,VERDANT OASIS PLAZA,UNITY MARCH,The Unity March is taking place at Verdant Oasis Plaza
40,VERDANT OASIS PLAZA,TRIBUNE SPOTLIGHT,Tribune Spotlight is reporting on the Unity march taking place at Verdant Oasis Plaza
41,VERDANT OASIS PLAZA,BAILEY ASADI,Bailey Asadi is speaking at Verdant Oasis Plaza about the march
43,HARMONY ASSEMBLY,UNITY MARCH,Harmony Assembly is organizing the Unity March

Output:
{{
    "title": "Verdant Oasis Plaza and Unity March",
    "summary": "The community revolves around the Verdant Oasis Plaza, which is the location of the Unity March. The plaza has relationships with the Harmony Assembly, Unity March, and Tribune Spotlight, all of which are associated with the march event.",
    "rating": 5.0,
    "rating_explanation": "The impact severity rating is moderate due to the potential for unrest or conflict during the Unity March.",
    "findings": [
        {{
            "summary": "Verdant Oasis Plaza as the central location",
            "explanation": "Verdant Oasis Plaza is the central entity in this community, serving as the location for the Unity March. This plaza is the common link between all other entities, suggesting its significance in the community. The plaza's association with the march could potentially lead to issues such as public disorder or conflict, depending on the nature of the march and the reactions it provokes. [Data: Entities (5), Relationships (37, 38, 39, 40, 41,+more)]"
        }},
        {{
            "summary": "Harmony Assembly's role in the community",
            "explanation": "Harmony Assembly is another key entity in this community, being the organizer of the march at Verdant Oasis Plaza. The nature of Harmony Assembly and its march could be a potential source of threat, depending on their objectives and the reactions they provoke. The relationship between Harmony Assembly and the plaza is crucial in understanding the dynamics of this community. [Data: Entities(6), Relationships (38, 43)]"
        }},
        {{
            "summary": "Unity March as a significant event",
            "explanation": "The Unity March is a significant event taking place at Verdant Oasis Plaza. This event is a key factor in the community's dynamics and could be a potential source of threat, depending on the nature of the march and the reactions it provokes. The relationship between the march and the plaza is crucial in understanding the dynamics of this community. [Data: Relationships (39)]"
        }},
        {{
            "summary": "Role of Tribune Spotlight",
            "explanation": "Tribune Spotlight is reporting on the Unity March taking place in Verdant Oasis Plaza. This suggests that the event has attracted media attention, which could amplify its impact on the community. The role of Tribune Spotlight could be significant in shaping public perception of the event and the entities involved. [Data: Relationships (40)]"
        }}
    ]
}}


# Real Data

Use the following text for your answer. Do not make anything up in your answer.

Text:
{input_text}

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- IMPACT SEVERITY RATING: a float score between 0-10 that represents the severity of IMPACT posed by entities within the community.  IMPACT is the scored importance of a community.
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
    {{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
        ]
    }}

# Grounding Rules

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (1), Entities (5, 7); Relationships (23); Claims (7, 2, 34, 64, 46, +more)]."

where 1, 5, 7, 23, 2, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.

Output:"""

COMMUNITY_REPORT_UPDATE_PROMPT = """
You are an AI assistant that helps a human analyst refine or update an existing community report based on additional entities, relationships, and optional claims. The goal is to enhance the completeness, accuracy, and relevance of the report while adhering to the structure and grounding rules provided below.

# Goal

Refine or update the provided report of a community by integrating new information from additional entities, relationships, and claims. The updated report should maintain a comprehensive overview of the community, highlight new insights, and reflect the potential impact of the added information. Ensure that the original insights are preserved unless contradicted or rendered obsolete by the new data.

# Report Structure

The updated report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title. If the new data introduces significant changes to the community’s key entities, adjust the title to reflect those changes. Otherwise, retain the existing title.
- SUMMARY: an executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities. Update the executive summary to reflect the added entities, relationships, and significant new information. 
- IMPACT SEVERITY RATING: If the new information affects the severity of the community’s impact, update the float score between 0-10 to represent the new level of IMPACT posed by entities within the community. IMPACT is the scored importance of a community.
- RATING EXPLANATION: Provide a single-sentence explanation in the IMPACT severity rating.
- DETAILED FINDINGS: Update the list of 5-10 key insights. Retain the original insights unless superseded or significantly altered by the new data. Incorporate new insights where appropriate. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return the updated report as a well-formed JSON-formatted string in the following format:
	{{
        "title": <report_title>,
        "summary": <executive_summary>,
        "rating": <impact_severity_rating>,
        "rating_explanation": <rating_explanation>,
        "findings": [
            {{
                "summary":<insight_1_summary>,
                "explanation": <insight_1_explanation>
            }},
            {{
                "summary":<insight_2_summary>,
                "explanation": <insight_2_explanation>
            }}
        ]
    }}

# Grounding Rules

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:
"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (1), Entities (5, 7); Relationships (23); Claims (7, 2, 34, 64, 46, +more)]."

where 1, 5, 7, 23, 2, 34, 46, and 64 represent the id (not the index) of the relevant data record.

Do not include information where the supporting evidence for it is not provided.

Any insights or information from the original report should retain their original data references unless those references are invalidated by the new data.

# Example Input
-----------
Original Report:

# Transformer Mechanisms in NLP

The community explores critical components of Transformer-based NLP systems, focusing on queries, keys, values, and self-attention mechanisms.

## Queries in NLP

Queries identify relevant keys and values, driving the attention mechanism. They enhance models' ability to focus on important text parts, improving NLP performance. [records: Entities (10), Relationships (23, 48, 59)]

## Keys and Values

Keys act as identifiers, while values represent data. Together, they enable efficient text processing and are vital for Transformer scalability. [records: Entities (12, 13), Relationships (35, 45)]

## Self-Attention

Self-attention captures token relationships, enabling Transformers to process sequences in parallel. This mechanism supports long-range dependency modeling. [records: Entities (11), Relationships (40, 47, 60)]

New Data:

Entities

id,entity,description
50,CONTEXT WINDOW,Defines the scope of tokens processed by the Transformer in a single step
51,POSITIONAL ENCODING,Encodes positional information to distinguish tokens in a sequence
52,MASKED ATTENTION,Enables selective focus on specific tokens within a sequence

Relationships

id,source,target,description
70,CONTEXT WINDOW,TRANSFORMERS,Context window size determines the maximum sequence length a Transformer can process
71,POSITIONAL ENCODING,TOKENS,Positional encoding ensures the model understands token order
72,MASKED ATTENTION,SELF-ATTENTION,Masked attention restricts the scope of self-attention to focus on specific tokens

Output:

{
    "title": "Transformer Mechanisms in NLP",
    "summary": "This community examines Transformer-based NLP systems, including queries, keys, values, self-attention, context window, positional encoding, and masked attention, which optimize language understanding and sequence processing.",
    "rating": 8.0,
    "rating_explanation": "The inclusion of context window, positional encoding, and masked attention expands the community's technical scope and impact.",
    "findings": [
        {
            "summary": "Queries enable attention",
            "explanation": "Queries guide attention by identifying key-value pairs, improving the ability of NLP models to focus on relevant parts of the input. The role of queries is integral in modern self-attention mechanisms. [Data: Entities (10), Relationships (23, 48, 59)]"
        },
        {
            "summary": "Keys and values streamline processing",
            "explanation": "Keys serve as references, and values store data to enable efficient attention computations. This integration supports scalability in Transformers, particularly when handling complex queries. The inclusion of the context window enhances the scalability further by constraining the range of operations within manageable limits. [Data: Entities (12, 13, 50), Relationships (35, 45, 70)]"
        },
        {
            "summary": "Self-attention supports parallelism and adaptability",
            "explanation": "Self-attention allows Transformers to process sequences in parallel while modeling token relationships effectively. Masked attention builds on this by restricting focus to specific tokens, which is critical for tasks such as language generation and masked language modeling. [Data: Entities (11, 52), Relationships (40, 47, 60, 72)]"
        },
        {
            "summary": "Context window defines sequence length",
            "explanation": "The context window sets the maximum tokens a Transformer can process at once, crucial for handling long sequences and maintaining computational efficiency. [Data: Entities (50), Relationships (70)]"
        },
        {
            "summary": "Positional encoding distinguishes token order",
            "explanation": "Positional encoding helps Transformers understand token order, ensuring sequence coherence and improving tasks like parsing and generation. [Data: Entities (51), Relationships (71)]"
        }
    ]
}

# Real Data

Use the following text for your answer. Do not make anything up in your answer. Include only data explicitly supported by the input.

Original Report:
{original_report}

New Data:
{new_data}

Output:"""
