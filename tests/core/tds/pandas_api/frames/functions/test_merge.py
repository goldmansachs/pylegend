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

import json
from textwrap import dedent
import datetime

import pytest

from pylegend._typing import (
    PyLegendDict,
    PyLegendUnion,
)
from pylegend.core.tds.pandas_api.frames.pandas_api_tds_frame import PandasApiTdsFrame
from pylegend.core.tds.tds_column import PrimitiveTdsColumn
from pylegend.core.tds.tds_frame import FrameToSqlConfig, FrameToPureConfig
from pylegend.extensions.tds.pandas_api.frames.pandas_api_table_spec_input_frame import PandasApiTableSpecInputFrame
from tests.test_helpers import generate_pure_query_and_compile
from tests.test_helpers.test_legend_service_frames import simple_person_service_frame_pandas_api
from pylegend.core.request.legend_client import LegendClient
from pylegend.core.database.sql_to_string import (
    SqlToStringConfig,
    SqlToStringFormat
)


class TestMergeFunction:

    @pytest.fixture(autouse=True)
    def init_legend(self, legend_test_server: PyLegendDict[str, PyLegendUnion[int,]]) -> None:
        self.legend_client = LegendClient("localhost", legend_test_server["engine_port"], secure_http=False)

    def test_merge_on_type_errors(self) -> None:
        columns = [
            PrimitiveTdsColumn.integer_column("col1"),
            PrimitiveTdsColumn.string_column("col2"),
            PrimitiveTdsColumn.float_column("col3"),
            PrimitiveTdsColumn.float_column("col4"),
            PrimitiveTdsColumn.float_column("col5")
        ]
        frame: PandasApiTdsFrame = PandasApiTableSpecInputFrame(['test_schema', 'test_table'], columns)

        columns2 = [
            PrimitiveTdsColumn.integer_column("pol1"),
            PrimitiveTdsColumn.string_column("pol2"),
            PrimitiveTdsColumn.float_column("pol3")
        ]
        frame2: PandasApiTdsFrame = PandasApiTableSpecInputFrame(['test_schema', 'test_table_2'], columns2)

        # other_frame type error
        with pytest.raises(TypeError) as v:
            frame.merge(123, how="inner")
        assert v.value.args[0] == "Can only merge TdsFrame objects, a <class 'int'> was passed"

        # how type error
        with pytest.raises(TypeError) as v:
            frame.merge(frame2, how=123)
        assert v.value.args[0] == "'how' must be str, got <class 'int'>"

        # on type error
        with pytest.raises(TypeError) as v:
            frame.merge(frame2, how="inner", on=123)
        assert v.value.args[0] == "Passing 'on' as a <class 'int'> is not supported. Provide 'on' as a tuple instead."

        # on type error
        with pytest.raises(TypeError) as v:
            frame.merge(frame2, how="inner", on=['col1', 2])
        assert v.value.args[0] == "'on' must contain only str elements"

        # left_on type error
        with pytest.raises(TypeError) as v:
            frame.merge(frame2, how="inner", left_on={"a": 1}, right_on='col1')
        assert v.value.args[0] == "Passing 'left_on' as a <class 'dict'> is not supported. Provide 'left_on' as a tuple instead."

        # right_on type error
        with pytest.raises(TypeError) as v:
            frame.merge(frame2, how="inner", left_on='col1', right_on={1, 2})
        assert v.value.args[0] == "Passing 'right_on' as a <class 'set'> is not supported. Provide 'right_on' as a tuple instead."

        # suffixes type error
        with pytest.raises(TypeError) as v:
            frame.merge(frame2, how="inner", suffixes={"x", "y"})
        assert v.value.args[0] == "Passing 'suffixes' as <class 'set'>, is not supported. Provide 'suffixes' as a tuple instead."

        # suffixes value error
        with pytest.raises(ValueError) as v:
            frame.merge(frame2, how="inner", suffixes=('_x', '_y', '_z'))
        assert v.value.args[0] == "too many values to unpack (expected 2)"

    def test_merge_on_unsupported_errors(self) -> None:
        columns = [
            PrimitiveTdsColumn.integer_column("col1"),
            PrimitiveTdsColumn.string_column("col2"),
            PrimitiveTdsColumn.float_column("col3"),
            PrimitiveTdsColumn.float_column("col4"),
            PrimitiveTdsColumn.float_column("col5")
        ]
        frame: PandasApiTdsFrame = PandasApiTableSpecInputFrame(['test_schema', 'test_table'], columns)

        columns2 = [
            PrimitiveTdsColumn.integer_column("pol1"),
            PrimitiveTdsColumn.string_column("pol2"),
            PrimitiveTdsColumn.float_column("pol3")
        ]
        frame2: PandasApiTdsFrame = PandasApiTableSpecInputFrame(['test_schema', 'test_table_2'], columns2)

        # same frame merge unsupported
        with pytest.raises(NotImplementedError) as v:
            frame.merge(frame, how="inner")
        assert v.value.args[0] == "Merging the same TdsFrame is not supported yet"

        # left_index unsupported
        with pytest.raises(NotImplementedError) as v:
            frame.merge(frame2, how="inner", left_index=True)
        assert v.value.args[0] == "Merging on index is not supported yet in PandasApi merge function"

        # right_index unsupported
        with pytest.raises(NotImplementedError) as v:
            frame.merge(frame2, how="inner", right_index=True)
        assert v.value.args[0] == "Merging on index is not supported yet in PandasApi merge function"

        # sort unsupported
        with pytest.raises(NotImplementedError) as v:
            frame.merge(frame2, how="inner", sort=True)
        assert v.value.args[0] == "Sort parameter is not supported yet in PandasApi merge function"

        # indicator unsupported
        with pytest.raises(NotImplementedError) as v:
            frame.merge(frame2, how="inner", indicator=True)
        assert v.value.args[0] == "Indicator parameter is not supported yet in PandasApi merge function"

        # validate unsupported
        with pytest.raises(NotImplementedError) as v:
            frame.merge(frame2, how="inner", validate="one_to_one")
        assert v.value.args[0] == "Validate parameter is not supported yet in PandasApi merge function"

    def test_merge_on_validation_errors(self) -> None:
        columns = [
            PrimitiveTdsColumn.integer_column("col1"),
            PrimitiveTdsColumn.string_column("col2"),
            PrimitiveTdsColumn.float_column("col3"),
            PrimitiveTdsColumn.float_column("col4"),
            PrimitiveTdsColumn.float_column("col5")
        ]
        frame: PandasApiTdsFrame = PandasApiTableSpecInputFrame(['test_schema', 'test_table'], columns)

        columns2 = [
            PrimitiveTdsColumn.integer_column("pol1"),
            PrimitiveTdsColumn.string_column("pol2"),
            PrimitiveTdsColumn.float_column("pol3")
        ]
        frame2: PandasApiTdsFrame = PandasApiTableSpecInputFrame(['test_schema', 'test_table_2'], columns2)

        # on and left_on/right_on both provided
        with pytest.raises(ValueError) as v:
            frame.merge(frame2, how="inner", on='col1', left_on='col1', right_on='pol1')
        assert v.value.args[0] == 'Can only pass argument "on" OR "left_on" and "right_on", not a combination of both.'

        # on key error
        with pytest.raises(KeyError) as v:
            frame.merge(frame2, how="inner", on='nol')
        assert v.value.args[0] == "'nol' not found"

        # left_on key error
        with pytest.raises(KeyError) as v:
            frame.merge(frame2, how="inner", left_on='nol', right_on='pol1')
        assert v.value.args[0] == "'nol' not found"

        # right_on key error
        with pytest.raises(KeyError) as v:
            frame.merge(frame2, how="inner", left_on='col1', right_on='nol')
        assert v.value.args[0] == "'nol' not found"

        # left_on and right_on length mismatch
        with pytest.raises(ValueError) as v:
            frame.merge(frame2, how="inner", left_on=['col1', 'col2'], right_on='pol1')
        assert v.value.args[0] == "len(right_on) must equal len(left_on)"

        # no resolution specified
        with pytest.raises(ValueError) as v:
            frame.merge(frame2, how="inner")
        assert v.value.args[0] == "No merge keys resolved. Specify 'on' or 'left_on'/'right_on', or ensure common columns."

    def test_merge_on_parameter(self) -> None:
        columns = [
            PrimitiveTdsColumn.integer_column("col1"),
            PrimitiveTdsColumn.string_column("col2"),
            PrimitiveTdsColumn.float_column("col3"),
            PrimitiveTdsColumn.float_column("col4"),
            PrimitiveTdsColumn.float_column("col5")
        ]
        frame: PandasApiTdsFrame = PandasApiTableSpecInputFrame(['test_schema', 'test_table'], columns)

        columns2 = [
            PrimitiveTdsColumn.integer_column("col1"),
            PrimitiveTdsColumn.string_column("pol2"),
            PrimitiveTdsColumn.float_column("pol3")
        ]
        frame2: PandasApiTdsFrame = PandasApiTableSpecInputFrame(['test_schema', 'test_table_2'], columns2)

        frame3: PandasApiTdsFrame = PandasApiTableSpecInputFrame(['test_schema', 'test_table_3'], columns2)

        # Inner
        merged_frame = frame.merge(frame2, how="inner", on='col1')

        expected = '''\
        SELECT
            "root"."col1" AS "col1",
            "root"."col2" AS "col2",
            "root"."col3" AS "col3",
            "root"."col4" AS "col4",
            "root"."col5" AS "col5",
            "root"."pol2" AS "pol2",
            "root"."pol3" AS "pol3"
        FROM
            (
                SELECT
                    "left"."col1" AS "col1",
                    "left"."col2" AS "col2",
                    "left"."col3" AS "col3",
                    "left"."col4" AS "col4",
                    "left"."col5" AS "col5",
                    "right"."pol2" AS "pol2",
                    "right"."pol3" AS "pol3"
                FROM
                    (
                        SELECT
                            "root".col1 AS "col1",
                            "root".col2 AS "col2",
                            "root".col3 AS "col3",
                            "root".col4 AS "col4",
                            "root".col5 AS "col5"
                        FROM
                            test_schema.test_table AS "root"
                    ) AS "left"
                    INNER JOIN
                        (
                            SELECT
                                "root".col1 AS "col1",
                                "root".pol2 AS "pol2",
                                "root".pol3 AS "pol3"
                            FROM
                                test_schema.test_table_2 AS "root"
                        ) AS "right"
                        ON ("left"."col1" = "right"."col1")
            ) AS "root"'''
        assert merged_frame.to_sql_query(FrameToSqlConfig()) == dedent(expected)
        assert generate_pure_query_and_compile(merged_frame, FrameToPureConfig(), self.legend_client) == dedent(
            '''\
              #Table(test_schema.test_table)#
                ->join(
                  #Table(test_schema.test_table_2)#
                    ->rename(
                      ~col1, ~col1__right_key_tmp
                    ),
                  JoinKind.INNER,
                  {l, r | $l.col1 == $r.col1__right_key_tmp}
                )
                ->project(
                  ~[col1:x|$x.col1, col2:x|$x.col2, col3:x|$x.col3, col4:x|$x.col4, col5:x|$x.col5, pol2:x|$x.pol2, pol3:x|$x.pol3]
                )'''
        )
        assert generate_pure_query_and_compile(merged_frame, FrameToPureConfig(pretty=False), self.legend_client) == \
               ("#Table(test_schema.test_table)#"
                "->join(#Table(test_schema.test_table_2)#->rename(~col1, ~col1__right_key_tmp), "
                "JoinKind.INNER, {l, r | $l.col1 == $r.col1__right_key_tmp})"
                "->project(~[col1:x|$x.col1, col2:x|$x.col2, col3:x|$x.col3, col4:x|$x.col4, "
                "col5:x|$x.col5, pol2:x|$x.pol2, pol3:x|$x.pol3])")

        # Left with suffix
        merged_frame = frame.merge(frame2, how="left", on='col1', suffixes=('_left', '_right'))

        expected = '''\
        SELECT
            "root"."col1" AS "col1",
            "root"."col2" AS "col2",
            "root"."col3" AS "col3",
            "root"."col4" AS "col4",
            "root"."col5" AS "col5",
            "root"."pol2" AS "pol2",
            "root"."pol3" AS "pol3"
        FROM
            (
                SELECT
                    "left"."col1" AS "col1",
                    "left"."col2" AS "col2",
                    "left"."col3" AS "col3",
                    "left"."col4" AS "col4",
                    "left"."col5" AS "col5",
                    "right"."pol2" AS "pol2",
                    "right"."pol3" AS "pol3"
                FROM
                    (
                        SELECT
                            "root".col1 AS "col1",
                            "root".col2 AS "col2",
                            "root".col3 AS "col3",
                            "root".col4 AS "col4",
                            "root".col5 AS "col5"
                        FROM
                            test_schema.test_table AS "root"
                    ) AS "left"
                    LEFT OUTER JOIN
                        (
                            SELECT
                                "root".col1 AS "col1",
                                "root".pol2 AS "pol2",
                                "root".pol3 AS "pol3"
                            FROM
                                test_schema.test_table_2 AS "root"
                        ) AS "right"
                        ON ("left"."col1" = "right"."col1")
            ) AS "root"'''
        assert merged_frame.to_sql_query(FrameToSqlConfig()) == dedent(expected)
        assert generate_pure_query_and_compile(merged_frame, FrameToPureConfig(), self.legend_client) == dedent(
            '''\
              #Table(test_schema.test_table)#
                ->join(
                  #Table(test_schema.test_table_2)#
                    ->rename(
                      ~col1, ~col1__right_key_tmp
                    ),
                  JoinKind.LEFT,
                  {l, r | $l.col1 == $r.col1__right_key_tmp}
                )
                ->project(
                  ~[col1:x|$x.col1, col2:x|$x.col2, col3:x|$x.col3, col4:x|$x.col4, col5:x|$x.col5, pol2:x|$x.pol2, pol3:x|$x.pol3]
                )'''
        )
        assert generate_pure_query_and_compile(merged_frame, FrameToPureConfig(pretty=False), self.legend_client) == \
               ("#Table(test_schema.test_table)#"
                "->join(#Table(test_schema.test_table_2)#->rename(~col1, ~col1__right_key_tmp), "
                "JoinKind.LEFT, {l, r | $l.col1 == $r.col1__right_key_tmp})"
                "->project(~[col1:x|$x.col1, col2:x|$x.col2, col3:x|$x.col3, col4:x|$x.col4, "
                "col5:x|$x.col5, pol2:x|$x.pol2, pol3:x|$x.pol3])")

        # # Right
        # merged_frame = frame2.merge(frame3, how="right")
        # expected = '''\
        #         SELECT
        #             "root"."col1" AS "col1",
        #             "root"."pol2" AS "pol2",
        #             "root"."pol3" AS "pol3"
        #         FROM
        #             (
        #                 SELECT
        #                     "left"."col1" AS "col1",
        #                     "left"."pol2" AS "pol2",
        #                     "left"."pol3" AS "pol3"
        #                 FROM
        #                     (
        #                         SELECT
        #                             "root".col1 AS "col1",
        #                             "root".pol2 AS "pol2",
        #                             "root".pol3 AS "pol3"
        #                         FROM
        #                             test_schema.test_table_2 AS "root"
        #                     ) AS "left"
        #                     RIGHT OUTER JOIN
        #                         (
        #                             SELECT
        #                                 "root".col1 AS "col1",
        #                                 "root".pol2 AS "pol2",
        #                                 "root".pol3 AS "pol3"
        #                             FROM
        #                                 test_schema.test_table_3 AS "root"
        #                         ) AS "right"
        #                         ON ((("left"."col1" = "right"."col1") AND ("left"."pol2" = "right"."pol2")) AND ("left"."pol3" = "right"."pol3"))
        #             ) AS "root"'''
        # assert merged_frame.to_sql_query(FrameToSqlConfig()) == dedent(expected)
        # print("PURE: ", merged_frame.to_pure_query())
        # assert generate_pure_query_and_compile(merged_frame, FrameToPureConfig(), self.legend_client) == dedent(
        #     '''\
        #       #Table(test_schema.test_table_2)#
        #         ->join(
        #           #Table(test_schema.test_table_3)#
        #             ->rename(
        #               ~col1, ~col1__right_key_tmp
        #             )
        #             ->rename(
        #               ~pol2, ~pol2__right_key_tmp
        #             )
        #             ->rename(
        #               ~pol3, ~pol3__right_key_tmp
        #             ),
        #           JoinKind.RIGHT,
        #           {l, r | $l.col1 == $r.col1__right_key_tmp && $l.pol2 == $r.pol2__right_key_tmp && $l.pol3 == $r.pol3__right_key_tmp}
        #         )
        #         ->project(
        #           ~[col1:x|$x.col1, pol2:x|$x.pol2, pol3:x|$x.pol3]
        #         )'''
        # )
        # assert generate_pure_query_and_compile(merged_frame, FrameToPureConfig(pretty=False), self.legend_client) == (
        #     "#Table(test_schema.test_table_2)#"
        #     "->join(#Table(test_schema.test_table_3)#"
        #     "->rename(~col1, ~col1__right_key_tmp)"
        #     "->rename(~pol2, ~pol2__right_key_tmp)"
        #     "->rename(~pol3, ~pol3__right_key_tmp), "
        #     "JoinKind.RIGHT, {l, r | $l.col1 == $r.col1__right_key_tmp && $l.pol2 == $r.pol2__right_key_tmp && $l.pol3 == $r.pol3__right_key_tmp})"
        #     "->project(~[col1:x|$x.col1, pol2:x|$x.pol2, pol3:x|$x.pol3])"
        # )

        # Full with suffix
        merged_frame = frame2.merge(frame3, on=['col1', 'pol2'], how="outer", suffixes=('_left', '_right'))
        expected = '''\
                SELECT
                    "root"."col1" AS "col1",
                    "root"."pol2" AS "pol2",
                    "root"."pol3_left" AS "pol3_left",
                    "root"."pol3_right" AS "pol3_right"
                FROM
                    (
                        SELECT
                            "left"."col1" AS "col1",
                            "left"."pol2" AS "pol2",
                            "left"."pol3" AS "pol3_left",
                            "right"."pol3" AS "pol3_right"
                        FROM
                            (
                                SELECT
                                    "root".col1 AS "col1",
                                    "root".pol2 AS "pol2",
                                    "root".pol3 AS "pol3"
                                FROM
                                    test_schema.test_table_2 AS "root"
                            ) AS "left"
                            FULL OUTER JOIN
                                (
                                    SELECT
                                        "root".col1 AS "col1",
                                        "root".pol2 AS "pol2",
                                        "root".pol3 AS "pol3"
                                    FROM
                                        test_schema.test_table_3 AS "root"
                                ) AS "right"
                                ON (("left"."col1" = "right"."col1") AND ("left"."pol2" = "right"."pol2"))
                    ) AS "root"'''
        assert merged_frame.to_sql_query(FrameToSqlConfig()) == dedent(expected)
        expected_pure_pretty = dedent(
            '''\
              #Table(test_schema.test_table_2)#
                ->rename(
                  ~pol3, ~pol3_left
                )
                ->join(
                  #Table(test_schema.test_table_3)#
                    ->rename(
                      ~pol3, ~pol3_right
                    )
                    ->rename(
                      ~col1, ~col1__right_key_tmp
                    )
                    ->rename(
                      ~pol2, ~pol2__right_key_tmp
                    ),
                  JoinKind.FULL,
                  {l, r | $l.col1 == $r.col1__right_key_tmp && $l.pol2 == $r.pol2__right_key_tmp}
                )
                ->project(
                  ~[col1:x|$x.col1, pol2:x|$x.pol2, pol3_left:x|$x.pol3_left, pol3_right:x|$x.pol3_right]
                )'''
        )
        assert generate_pure_query_and_compile(merged_frame, FrameToPureConfig(), self.legend_client) == expected_pure_pretty
        expected_pure_compact = (
            "#Table(test_schema.test_table_2)#"
            "->rename(~pol3, ~pol3_left)"
            "->join(#Table(test_schema.test_table_3)#"
            "->rename(~pol3, ~pol3_right)"
            "->rename(~col1, ~col1__right_key_tmp)"
            "->rename(~pol2, ~pol2__right_key_tmp), "
            "JoinKind.FULL, {l, r | $l.col1 == $r.col1__right_key_tmp && $l.pol2 == $r.pol2__right_key_tmp})"
            "->project(~[col1:x|$x.col1, pol2:x|$x.pol2, pol3_left:x|$x.pol3_left, pol3_right:x|$x.pol3_right])"
        )
        assert generate_pure_query_and_compile(merged_frame, FrameToPureConfig(pretty=False), self.legend_client) == expected_pure_compact

    def test_merge_left_right_on_parameters(self) -> None:
        columns = [
            PrimitiveTdsColumn.integer_column("col1"),
            PrimitiveTdsColumn.string_column("col2"),
            PrimitiveTdsColumn.float_column("col3"),
            PrimitiveTdsColumn.float_column("col4"),
            PrimitiveTdsColumn.float_column("col5")
        ]
        frame: PandasApiTdsFrame = PandasApiTableSpecInputFrame(['test_schema', 'test_table'], columns)

        columns2 = [
            PrimitiveTdsColumn.integer_column("col1"),
            PrimitiveTdsColumn.string_column("pol2"),
            PrimitiveTdsColumn.float_column("pol3")
        ]
        frame2: PandasApiTdsFrame = PandasApiTableSpecInputFrame(['test_schema', 'test_table_2'], columns2)

        # inner
        merged_frame = frame.merge(frame2, how="inner", left_on='col1', right_on='col1')
        expected = '''\
                SELECT
                    "root"."col1" AS "col1",
                    "root"."col2" AS "col2",
                    "root"."col3" AS "col3",
                    "root"."col4" AS "col4",
                    "root"."col5" AS "col5",
                    "root"."pol2" AS "pol2",
                    "root"."pol3" AS "pol3"
                FROM
                    (
                        SELECT
                            "left"."col1" AS "col1",
                            "left"."col2" AS "col2",
                            "left"."col3" AS "col3",
                            "left"."col4" AS "col4",
                            "left"."col5" AS "col5",
                            "right"."pol2" AS "pol2",
                            "right"."pol3" AS "pol3"
                        FROM
                            (
                                SELECT
                                    "root".col1 AS "col1",
                                    "root".col2 AS "col2",
                                    "root".col3 AS "col3",
                                    "root".col4 AS "col4",
                                    "root".col5 AS "col5"
                                FROM
                                    test_schema.test_table AS "root"
                            ) AS "left"
                            INNER JOIN
                                (
                                    SELECT
                                        "root".col1 AS "col1",
                                        "root".pol2 AS "pol2",
                                        "root".pol3 AS "pol3"
                                    FROM
                                        test_schema.test_table_2 AS "root"
                                ) AS "right"
                                ON ("left"."col1" = "right"."col1")
                    ) AS "root"'''
        assert merged_frame.to_sql_query(FrameToSqlConfig()) == dedent(expected)
        assert generate_pure_query_and_compile(merged_frame, FrameToPureConfig(), self.legend_client) == dedent(
            '''\
              #Table(test_schema.test_table)#
                ->join(
                  #Table(test_schema.test_table_2)#
                    ->rename(
                      ~col1, ~col1__right_key_tmp
                    ),
                  JoinKind.INNER,
                  {l, r | $l.col1 == $r.col1__right_key_tmp}
                )
                ->project(
                  ~[col1:x|$x.col1, col2:x|$x.col2, col3:x|$x.col3, col4:x|$x.col4, col5:x|$x.col5, pol2:x|$x.pol2, pol3:x|$x.pol3]
                )'''
        )
        assert generate_pure_query_and_compile(merged_frame, FrameToPureConfig(pretty=False), self.legend_client) == \
               ("#Table(test_schema.test_table)#"
                "->join(#Table(test_schema.test_table_2)#->rename(~col1, ~col1__right_key_tmp), "
                "JoinKind.INNER, {l, r | $l.col1 == $r.col1__right_key_tmp})"
                "->project(~[col1:x|$x.col1, col2:x|$x.col2, col3:x|$x.col3, col4:x|$x.col4, "
                "col5:x|$x.col5, pol2:x|$x.pol2, pol3:x|$x.pol3])")

        # left
        merged_frame = frame.merge(frame2, how="left", left_on='col1', right_on='pol3', suffixes=('_left', '_right'))
        expected = '''\
                SELECT
                    "root"."col1_left" AS "col1_left",
                    "root"."col2" AS "col2",
                    "root"."col3" AS "col3",
                    "root"."col4" AS "col4",
                    "root"."col5" AS "col5",
                    "root"."col1_right" AS "col1_right",
                    "root"."pol2" AS "pol2",
                    "root"."pol3" AS "pol3"
                FROM
                    (
                        SELECT
                            "left"."col1" AS "col1_left",
                            "left"."col2" AS "col2",
                            "left"."col3" AS "col3",
                            "left"."col4" AS "col4",
                            "left"."col5" AS "col5",
                            "right"."col1" AS "col1_right",
                            "right"."pol2" AS "pol2",
                            "right"."pol3" AS "pol3"
                        FROM
                            (
                                SELECT
                                    "root".col1 AS "col1",
                                    "root".col2 AS "col2",
                                    "root".col3 AS "col3",
                                    "root".col4 AS "col4",
                                    "root".col5 AS "col5"
                                FROM
                                    test_schema.test_table AS "root"
                            ) AS "left"
                            LEFT OUTER JOIN
                                (
                                    SELECT
                                        "root".col1 AS "col1",
                                        "root".pol2 AS "pol2",
                                        "root".pol3 AS "pol3"
                                    FROM
                                        test_schema.test_table_2 AS "root"
                                ) AS "right"
                                ON ("left"."col1" = "right"."pol3")
                    ) AS "root"'''
        assert merged_frame.to_sql_query(FrameToSqlConfig()) == dedent(expected)
        expected_pure_pretty = dedent(
            '''\
              #Table(test_schema.test_table)#
                ->rename(
                  ~col1, ~col1_left
                )
                ->join(
                  #Table(test_schema.test_table_2)#
                    ->rename(
                      ~col1, ~col1_right
                    ),
                  JoinKind.LEFT,
                  {l, r | $l.col1_left == $r.pol3}
                )'''
        )
        assert generate_pure_query_and_compile(merged_frame, FrameToPureConfig(), self.legend_client) == expected_pure_pretty
        expected_pure_compact = (
            "#Table(test_schema.test_table)#"
            "->rename(~col1, ~col1_left)"
            "->join(#Table(test_schema.test_table_2)#"
            "->rename(~col1, ~col1_right), "
            "JoinKind.LEFT, {l, r | $l.col1_left == $r.pol3})"
        )
        assert generate_pure_query_and_compile(merged_frame, FrameToPureConfig(pretty=False), self.legend_client) == expected_pure_compact

        # right
        merged_frame = frame.merge(frame2, how="right", left_on=['col1', 'col2'], right_on=['col1', 'pol2'])
        expected_sql = '''\
                SELECT
                    "root"."col1" AS "col1",
                    "root"."col2" AS "col2",
                    "root"."col3" AS "col3",
                    "root"."col4" AS "col4",
                    "root"."col5" AS "col5",
                    "root"."pol2" AS "pol2",
                    "root"."pol3" AS "pol3"
                FROM
                    (
                        SELECT
                            "left"."col1" AS "col1",
                            "left"."col2" AS "col2",
                            "left"."col3" AS "col3",
                            "left"."col4" AS "col4",
                            "left"."col5" AS "col5",
                            "right"."pol2" AS "pol2",
                            "right"."pol3" AS "pol3"
                        FROM
                            (
                                SELECT
                                    "root".col1 AS "col1",
                                    "root".col2 AS "col2",
                                    "root".col3 AS "col3",
                                    "root".col4 AS "col4",
                                    "root".col5 AS "col5"
                                FROM
                                    test_schema.test_table AS "root"
                            ) AS "left"
                            RIGHT OUTER JOIN
                                (
                                    SELECT
                                        "root".col1 AS "col1",
                                        "root".pol2 AS "pol2",
                                        "root".pol3 AS "pol3"
                                    FROM
                                        test_schema.test_table_2 AS "root"
                                ) AS "right"
                                ON (("left"."col1" = "right"."col1") AND ("left"."col2" = "right"."pol2"))
                    ) AS "root"'''

        assert merged_frame.to_sql_query(FrameToSqlConfig()) == dedent(expected_sql)
        expected_pure_pretty = dedent(
            '''\
              #Table(test_schema.test_table)#
                ->join(
                  #Table(test_schema.test_table_2)#
                    ->rename(
                      ~col1, ~col1__right_key_tmp
                    ),
                  JoinKind.RIGHT,
                  {l, r | $l.col1 == $r.col1__right_key_tmp && $l.col2 == $r.pol2}
                )
                ->project(
                  ~[col1:x|$x.col1, col2:x|$x.col2, col3:x|$x.col3, col4:x|$x.col4, col5:x|$x.col5, pol2:x|$x.pol2, pol3:x|$x.pol3]
                )'''
        )
        assert generate_pure_query_and_compile(merged_frame, FrameToPureConfig(), self.legend_client) == expected_pure_pretty
        expected_pure_compact = (
            "#Table(test_schema.test_table)#"
            "->join(#Table(test_schema.test_table_2)#"
            "->rename(~col1, ~col1__right_key_tmp), "
            "JoinKind.RIGHT, {l, r | $l.col1 == $r.col1__right_key_tmp && $l.col2 == $r.pol2})"
            "->project(~[col1:x|$x.col1, col2:x|$x.col2, col3:x|$x.col3, col4:x|$x.col4, col5:x|$x.col5, pol2:x|$x.pol2, pol3:x|$x.pol3])"
        )
        assert generate_pure_query_and_compile(merged_frame, FrameToPureConfig(pretty=False), self.legend_client) == expected_pure_compact

    def test_merge_chained(self) -> None:
        columns = [
            PrimitiveTdsColumn.integer_column("col1"),
            PrimitiveTdsColumn.string_column("col2"),
            PrimitiveTdsColumn.float_column("col3"),
            PrimitiveTdsColumn.float_column("col4"),
            PrimitiveTdsColumn.float_column("col5")
        ]
        frame: PandasApiTdsFrame = PandasApiTableSpecInputFrame(['test_schema', 'test_table'], columns)

        columns2 = [
            PrimitiveTdsColumn.integer_column("col1"),
            PrimitiveTdsColumn.string_column("pol2"),
            PrimitiveTdsColumn.float_column("pol3")
        ]
        frame2: PandasApiTdsFrame = PandasApiTableSpecInputFrame(['test_schema', 'test_table_2'], columns2)

        # Merge
        merged_frame = frame.merge(frame2, how="inner", on='col1').merge(frame2, how="left", left_on='col1', right_on='col1', suffixes=('_left', '_right'))

        # Truncate
        merged_frame = frame.merge(frame2)
        newframe = merged_frame.truncate(before = 1, after = 3)


    def test_e2e_merge(self, legend_test_server: PyLegendDict[str, PyLegendUnion[int,]]) -> None:
        frame: PandasApiTdsFrame = simple_person_service_frame_pandas_api(legend_test_server["engine_port"])
        frame2: PandasApiTdsFrame = simple_person_service_frame_pandas_api(legend_test_server["engine_port"])

        # on with suffix
        newframe = frame.merge(frame2, how = 'left', on=['First Name', 'Age'])
        expected = {
            "columns": [
                "First Name",
                "Last Name_x",
                "Age",
                "Firm/Legal Name_x",
                "Last Name_y",
                "Firm/Legal Name_y",
            ],
            "rows": [
                {"values": ["Peter", "Smith", 23, "Firm X", "Smith", "Firm X"]},
                {"values": ["John", "Johnson", 22, "Firm X", "Johnson", "Firm X"]},
                {"values": ["John", "Hill", 12, "Firm X", "Hill", "Firm X"]},
                {"values": ["Anthony", "Allen", 22, "Firm X", "Allen", "Firm X"]},
                {"values": ["Fabrice", "Roberts", 34, "Firm A", "Roberts", "Firm A"]},
                {"values": ["Oliver", "Hill", 32, "Firm B", "Hill", "Firm B"]},
                {"values": ["David", "Harris", 35, "Firm C", "Harris", "Firm C"]},
            ],
        }
        res = newframe.execute_frame_to_string()
        assert json.loads(res)["result"] == expected

        # Multiple columns
        # newframe = frame[['First Name', 'Age']]
        # expected = {
        #     "columns": ["First Name", "Age"],
        #     "rows": [
        #         {"values": ["Peter", 23]},
        #         {"values": ["John", 22]},
        #         {"values": ["John", 12]},
        #         {"values": ["Anthony", 22]},
        #         {"values": ["Fabrice", 34]},
        #         {"values": ["Oliver", 32]},
        #         {"values": ["David", 35]},
        #     ],
        # }
        # res = newframe.execute_frame_to_string()
        # assert json.loads(res)["result"] == expected
