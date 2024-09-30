# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing create_community_reports and load_strategy methods definition."""

import logging
from enum import Enum
from typing import cast

import pandas as pd
from datashaper import (
    AsyncType,
    NoopVerbCallbacks,
    TableContainer,
    VerbCallbacks,
    VerbInput,
    derive_from_rows,
    progress_ticker,
    verb,
)

import graphrag.config.defaults as defaults
import graphrag.index.graph.extractors.community_reports.schemas as schemas
from graphrag.index.cache import PipelineCache
from graphrag.index.graph.extractors.community_reports import (
    get_levels,
    prep_community_report_context,
)
from graphrag.index.utils.ds_util import get_required_input_table

from .strategies.typing import CommunityReport, CommunityReportsStrategy

log = logging.getLogger(__name__)


class CreateCommunityReportsStrategyType(str, Enum):
    """CreateCommunityReportsStrategyType class definition."""

    graph_intelligence = "graph_intelligence"

    def __repr__(self):
        """Get a string representation."""
        return f'"{self.value}"'


@verb(name="create_community_reports")
async def create_community_reports(
    input: VerbInput,
    callbacks: VerbCallbacks,
    cache: PipelineCache,
    strategy: dict,
    async_mode: AsyncType = AsyncType.AsyncIO,
    num_threads: int = 4,
    **_kwargs,
) -> TableContainer:
    """Generate entities for each row, and optionally a graph of those entities."""
    log.debug("create_community_reports strategy=%s", strategy)
    local_contexts = cast(pd.DataFrame, input.get_input())
    nodes_ctr = get_required_input_table(input, "nodes")
    nodes = cast(pd.DataFrame, nodes_ctr.table)
    community_hierarchy_ctr = get_required_input_table(input, "community_hierarchy")
    community_hierarchy = cast(pd.DataFrame, community_hierarchy_ctr.table)

    levels = get_levels(nodes)
    reports: list[CommunityReport | None] = []
    tick = progress_ticker(callbacks.progress, len(local_contexts))
    runner = load_strategy(strategy["type"])
    
    
    """
    local_contexts df (max_input_length=1000):
    all_contexts is a list of each entity's description in "node_details" along with incident edges in "edge_details"
     community                                        all_context                                     context_string  context_size  context_exceed_limit  level
0        10  [{'title': 'ALPACA', 'degree': 2, 'node_detail...  -----Entities-----\nhuman_readable_id,title,de...          1550                  True      2
1         9  [{'title': 'LLM INFERENCE', 'degree': 1, 'node...  -----Entities-----\nhuman_readable_id,title,de...           173                 False      2
0         4  [{'title': 'ALPACA', 'degree': 2, 'node_detail...  -----Entities-----\nhuman_readable_id,title,de...          1673                  True      1
1         5  [{'title': 'FASTER TRANSFORMER', 'degree': 3, ...  -----Entities-----\nhuman_readable_id,title,de...           416                 False      1
2         6  [{'title': 'GPU', 'degree': 2, 'node_details':...  -----Entities-----\nhuman_readable_id,title,de...           321                 False      1
3         7  [{'title': 'FASTERTRANSFORMER', 'degree': 7, '...  -----Entities-----\nhuman_readable_id,title,de...          1706                  True      1
4         8  [{'title': 'ORCA', 'degree': 8, 'node_details'...  -----Entities-----\nhuman_readable_id,title,de...           327                 False      1
0         0  [{'title': 'ALPACA', 'degree': 2, 'node_detail...  -----Entities-----\nhuman_readable_id,title,de...          2146                  True      0
1         1  [{'title': 'COPY-ON-WRITE', 'degree': 2, 'node...  -----Entities-----\nhuman_readable_id,title,de...           793                 False      0
2         2  [{'title': 'GPT', 'degree': 2, 'node_details':...  -----Entities-----\nhuman_readable_id,title,de...           277                 False      0
3         3  [{'title': 'FASTERTRANSFORMER', 'degree': 7, '...  -----Entities-----\nhuman_readable_id,title,de...          1832                  True      0
    
    example entity in all_contexts:

    {
        "title": "OPT-13B",
        "degree": 1,
        "node_details": {
            "human_readable_id": 411,
            "title": "OPT-13B",
            "description": "OPT-13B is a type of computer model that has demonstrated exceptional processing capabilities. It can handle a significant number of requests simultaneously, outperforming other models such as Orca (Oracle) and Orca (Max). Specifically, OPT-13B can process 2.2 times more requests at the same time than Orca (Oracle) and 4.3 times more requests than Orca (Max), showcasing its impressive scalability and efficiency.",
            "degree": 1,
        },
        "edge_details": [
            nan,
            {
                "human_readable_id": "62",
                "source": "VLLM",
                "target": "OPT-13B",
                "description": "vLLM can process 2.2× more requests at the same time than Orca (Oracle) and 4.3× more requests than Orca (Max).",
                "rank": 30,
            },
        ],
        "claim_details": [],
    }

    """
    
    # 1. reverse levels to start from top level
    # 2. make report solely based on local context
    # 3. if local context exceeds the limit, trim context based on degrees or ranking algorithms
    pruning_strategy = strategy.get("local_context_pruning_strategy", "none")
    if pruning_strategy == "degree":
        levels = list(reversed(levels))
    
    for level in levels:    # level 0 is root level
        time_start = pd.Timestamp.now()
        level_contexts = prep_community_report_context(
            pd.DataFrame(reports),
            local_context_df=local_contexts,
            community_hierarchy_df=community_hierarchy,
            level=level,
            max_tokens=strategy.get(
                # "max_input_tokens", defaults.COMMUNITY_REPORT_MAX_INPUT_LENGTH
                "max_input_length", defaults.COMMUNITY_REPORT_MAX_INPUT_LENGTH
            ),
            pruning_strategy=pruning_strategy,
        )
        
        async def run_generate(record):
            result = await _generate_report(
                runner,
                community_id=record[schemas.NODE_COMMUNITY],
                community_level=record[schemas.COMMUNITY_LEVEL],
                community_context=record[schemas.CONTEXT_STRING],
                cache=cache,
                callbacks=callbacks,
                strategy=strategy,
            )
            tick()
            return result

        local_reports = await derive_from_rows(
            level_contexts,
            run_generate,
            callbacks=NoopVerbCallbacks(),
            num_threads=num_threads,
            scheduling_type=async_mode,
        )
        reports.extend([lr for lr in local_reports if lr is not None])
        time_end = pd.Timestamp.now()
        log.debug("[CR_GEN TIME] Level %s community report generation took %s", level, time_end - time_start)
    return TableContainer(table=pd.DataFrame(reports))


async def _generate_report(
    runner: CommunityReportsStrategy,
    cache: PipelineCache,
    callbacks: VerbCallbacks,
    strategy: dict,
    community_id: int | str,
    community_level: int,
    community_context: str,
) -> CommunityReport | None:
    """Generate a report for a single community."""
    return await runner(
        community_id, community_context, community_level, callbacks, cache, strategy
    )


def load_strategy(
    strategy: CreateCommunityReportsStrategyType,
) -> CommunityReportsStrategy:
    """Load strategy method definition."""
    match strategy:
        case CreateCommunityReportsStrategyType.graph_intelligence:
            from .strategies.graph_intelligence import run

            return run
        case _:
            msg = f"Unknown strategy: {strategy}"
            raise ValueError(msg)
