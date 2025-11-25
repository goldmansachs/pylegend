# Copyright 2025 Goldman Sachs
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pylegend._typing import (
    PyLegendList,
    PyLegendSequence,
    PyLegendUnion,
    PyLegendOptional,
    PyLegendTuple,
)
from pylegend.core.tds.pandas_api.frames.pandas_api_applied_function_tds_frame import PandasApiAppliedFunction
from pylegend.core.tds.pandas_api.frames.pandas_api_base_tds_frame import PandasApiBaseTdsFrame
from pylegend.core.tds.tds_column import TdsColumn
from pylegend.core.tds.tds_frame import FrameToSqlConfig, FrameToPureConfig
from pylegend.core.tds.sql_query_helpers import copy_query, create_sub_query, extract_columns_for_subquery
from pylegend.core.sql.metamodel import (
    QuerySpecification,
    Select,
    SelectItem,
    SingleColumn,
    AliasedRelation,
    TableSubquery,
    Query,
    Join,
    JoinType,
    JoinOn,
    QualifiedNameReference,
    QualifiedName,
)

from pylegend.core.language.pandas_api.pandas_api_tds_row import PandasApiTdsRow
from pylegend.core.language import (
    PyLegendBoolean,
    PyLegendBooleanLiteralExpression,
    PyLegendPrimitive,
)
from pylegend.core.language.shared.helpers import generate_pure_lambda

__all__: PyLegendSequence[str] = [
    "PandasApiMergeFunction"
]


class PandasApiMergeFunction(PandasApiAppliedFunction):
    __base_frame: PandasApiBaseTdsFrame
    __other_frame: PandasApiBaseTdsFrame
    __on: PyLegendOptional[PyLegendUnion[str, PyLegendList[str]]]
    __left_on: PyLegendOptional[PyLegendUnion[str, PyLegendList[str]]]
    __right_on: PyLegendOptional[PyLegendUnion[str, PyLegendList[str]]]
    __how: PyLegendOptional[str]
    __suffixes: PyLegendOptional[PyLegendTuple[str, str]]

    @classmethod
    def name(cls) -> str:
        return "merge"  # pragma: no cover

    def __init__(
        self,
        base_frame: PandasApiBaseTdsFrame,
        other_frame: PandasApiBaseTdsFrame,
        on: PyLegendOptional[PyLegendUnion[str, PyLegendList[str]]],
        left_on: PyLegendOptional[PyLegendUnion[str, PyLegendList[str]]],
        right_on: PyLegendOptional[PyLegendUnion[str, PyLegendList[str]]],
        how: PyLegendOptional[str],
        suffixes: PyLegendOptional[PyLegendTuple[str, str]]
    ) -> None:
        if not isinstance(other_frame, PandasApiBaseTdsFrame):
            raise ValueError("Expected PandasApiBaseTdsFrame for 'other'")  # pragma: no cover
        self.__base_frame = base_frame
        self.__other_frame = other_frame
        self.__on = on
        self.__left_on = left_on
        self.__right_on = right_on
        self.__how = how
        self.__suffixes = suffixes

    # Key resolution helpers
    def __normalize_keys(
        self,
        candidate: PyLegendUnion[str, PyLegendList[str], None]
    ) -> PyLegendList[str]:
        if candidate is None:
            return []
        return [candidate] if isinstance(candidate, str) else list(candidate)

    def __derive_key_pairs(self) -> PyLegendList[PyLegendTuple[str, str]]:
        left_cols = [c.get_name() for c in self.__base_frame.columns()]
        right_cols = [c.get_name() for c in self.__other_frame.columns()]

        if self.__on is not None and (self.__left_on is not None or self.__right_on is not None):
            raise ValueError("Cannot pass 'on' together with 'left_on' or 'right_on'")
        if self.__on is not None:
            on_keys = self.__normalize_keys(self.__on)
            for k in on_keys:
                if k not in left_cols or k not in right_cols:
                    raise ValueError(f"Merge key '{k}' not present in both frames")
            return [(k, k) for k in on_keys]

        left_keys = self.__normalize_keys(self.__left_on)
        right_keys = self.__normalize_keys(self.__right_on)
        if left_keys or right_keys:
            if len(left_keys) != len(right_keys):
                raise ValueError("left_on and right_on must be same length")
            for lk in left_keys:
                if lk not in left_cols:
                    raise ValueError(f"Left key '{lk}' not found")
            for rk in right_keys:
                if rk not in right_cols:
                    raise ValueError(f"Right key '{rk}' not found")
            return list(zip(left_keys, right_keys))

        # Infer intersection
        inferred = list(set(left_cols) & set(right_cols))
        return [(k, k) for k in inferred]

    # Internal auto join condition builder (returns PyLegendBoolean expression)
    def __build_condition(self) -> PyLegendPrimitive:
        key_pairs = self.__derive_key_pairs()
        left_row = PandasApiTdsRow.from_tds_frame("left", self.__base_frame)
        right_row = PandasApiTdsRow.from_tds_frame("right", self.__other_frame)
        if not key_pairs:
            return PyLegendBoolean(PyLegendBooleanLiteralExpression(True))
        expr = None
        for l, r in key_pairs:
            part = (left_row[l] == right_row[r])
            expr = part if expr is None else (expr & part)
        return expr  # type: ignore

    def __join_type(self) -> JoinType:
        how_lower = self.__how.lower()
        if how_lower == "inner":
            return JoinType.INNER
        if how_lower in ("left", "left_outer"):
            return JoinType.LEFT
        if how_lower in ("right", "right_outer"):
            return JoinType.RIGHT
        if how_lower in ("outer", "full", "full_outer"):
            return JoinType.FULL
        raise ValueError("Unsupported merge how - " + self.__how)

    def to_sql(self, config: FrameToSqlConfig) -> QuerySpecification:
        db_extension = config.sql_to_string_generator().get_db_extension()
        left_query = copy_query(self.__base_frame.to_sql_query_object(config))
        right_query = copy_query(self.__other_frame.to_sql_query_object(config))

        join_condition_expr = self.__build_condition()
        if isinstance(join_condition_expr, bool):  # safety
            join_condition_expr = PyLegendBoolean(PyLegendBooleanLiteralExpression(join_condition_expr))
        join_sql_expr = join_condition_expr.to_sql_expression(
            {
                "left": create_sub_query(left_query, config, "left"),
                "right": create_sub_query(right_query, config, "right"),
            },
            config
        )

        left_alias = db_extension.quote_identifier("left")
        right_alias = db_extension.quote_identifier("right")

        # Recalculate columns (adds suffixes)
        result_columns = self.calculate_columns()
        # Need mapping from original columns to aliases referencing left/right
        left_original = {c.get_name(): c for c in self.__base_frame.columns()}
        right_original = {c.get_name(): c for c in self.__other_frame.columns()}
        key_pairs = self.__derive_key_pairs()
        same_name_keys = {l for l, r in key_pairs if l == r}

        select_items: PyLegendList[SelectItem] = []
        # Left select items (with potential renamed overlapping)
        left_cols_set = set(left_original.keys())
        right_cols_set = set(right_original.keys())
        overlapping = (left_cols_set & right_cols_set) - same_name_keys

        for c in self.__base_frame.columns():
            orig = c.get_name()
            out_name = orig + self.__suffixes[0] if orig in overlapping else orig
            q_out = db_extension.quote_identifier(out_name)
            q_in = db_extension.quote_identifier(orig)
            select_items.append(
                SingleColumn(q_out, QualifiedNameReference(QualifiedName(parts=[left_alias, q_in])))
            )

        # Right select items
        for c in self.__other_frame.columns():
            orig = c.get_name()
            # Skip same-name join key
            if orig in same_name_keys:
                continue
            out_name = orig + self.__suffixes[1] if orig in overlapping else orig
            q_out = db_extension.quote_identifier(out_name)
            q_in = db_extension.quote_identifier(orig)
            select_items.append(
                SingleColumn(q_out, QualifiedNameReference(QualifiedName(parts=[right_alias, q_in])))
            )

        join_spec = QuerySpecification(
            select=Select(selectItems=select_items, distinct=False),
            from_=[
                Join(
                    type_=self.__join_type(),
                    left=AliasedRelation(
                        relation=TableSubquery(Query(queryBody=left_query, limit=None, offset=None, orderBy=[])),
                        alias=left_alias,
                        columnNames=extract_columns_for_subquery(left_query)
                    ),
                    right=AliasedRelation(
                        relation=TableSubquery(Query(queryBody=right_query, limit=None, offset=None, orderBy=[])),
                        alias=right_alias,
                        columnNames=extract_columns_for_subquery(right_query)
                    ),
                    criteria=JoinOn(expression=join_sql_expr)
                )
            ],
            where=None,
            groupBy=[],
            having=None,
            orderBy=[],
            limit=None,
            offset=None
        )
        return create_sub_query(join_spec, config, "root")

    def to_pure(self, config: FrameToPureConfig) -> str:
        # Represented as a join with auto lambda
        join_kind = (
            "INNER" if self.__how.lower() == "inner" else
            "LEFT" if self.__how.lower() in ("left", "left_outer") else
            "RIGHT" if self.__how.lower() in ("right", "right_outer") else
            "FULL"
        )
        left_row = PandasApiTdsRow.from_tds_frame("l", self.__base_frame)
        right_row = PandasApiTdsRow.from_tds_frame("r", self.__other_frame)
        cond_expr = self.__build_condition()
        cond_str = cond_expr.to_pure_expression(config.push_indent(2))
        return (f"{self.__base_frame.to_pure(config)}{config.separator(1)}"
                f"->merge({config.separator(2)}"
                f"{self.__other_frame.to_pure(config.push_indent(2))},{config.separator(2, True)}"
                f"JoinKind.{join_kind},{config.separator(2, True)}"
                f"{generate_pure_lambda('l, r', cond_str)}{config.separator(1)})")

    def base_frame(self) -> PandasApiBaseTdsFrame:
        return self.__base_frame

    def tds_frame_parameters(self) -> PyLegendList[PandasApiBaseTdsFrame]:
        return [self.__other_frame]

    # Columns after merge
    def calculate_columns(self) -> PyLegendSequence[TdsColumn]:
        key_pairs = self.__derive_key_pairs()
        left_keys_same_name = {l for l, r in key_pairs if l == r}
        left_cols = [c.get_name() for c in self.__base_frame.columns()]
        right_cols = [c.get_name() for c in self.__other_frame.columns()]

        overlapping = (set(left_cols) & set(right_cols)) - left_keys_same_name
        # Build left columns (apply suffix to overlapping non-key)
        result_cols: PyLegendList[TdsColumn] = []
        for c in self.__base_frame.columns():
            name = c.get_name()
            if name in overlapping:
                result_cols.append(TdsColumn(name + self.__suffixes[0], c.get_type()))
            else:
                result_cols.append(c.copy())

        # Add right columns (skip same-name keys; apply suffix to overlapping non-key)
        for c in self.__other_frame.columns():
            name = c.get_name()
            if any(name == l and l == r for l, r in key_pairs):
                continue  # key appears already
            if name in overlapping:
                result_cols.append(TdsColumn(name + self.__suffixes[1], c.get_type()))
            else:
                result_cols.append(c.copy())

        # Validate no duplicates
        names = [c.get_name() for c in result_cols]
        if len(names) != len(set(names)):
            raise ValueError("Resulting merged columns contain duplicates after suffix application")
        return result_cols

    def validate(self) -> bool:
        # Suffix validation
        if not (isinstance(self.__suffixes, tuple) and len(self.__suffixes) == 2):
            raise ValueError("suffixes must be a tuple of length 2")
        self.__derive_key_pairs()  # runs key validations
        self.__join_type()         # validates how
        # Column duplication (handled later) but we can still check ambiguous scenario:
        return True
