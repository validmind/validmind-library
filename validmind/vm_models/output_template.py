# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

from datetime import datetime

from dateutil import parser
from jinja2 import Environment


def format_date(value, format="%Y-%m-%d"):
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.strftime(format)

    return parser.parse(value).strftime(format)


def format_number(value, format="{:,.4f}"):
    return format.format(value)


def _generate_empty_historical_data(value):
    return [
        {
            "value": value,
            "metadata": {
                "created_at": format_date(datetime.now()),
            },
        }
    ]


class OutputTemplate:
    def __init__(self, template_string, template_engine=None):
        if template_engine is None:
            template_engine = Environment()
            template_engine.filters["date"] = format_date
            template_engine.filters["number"] = format_number

        self.template_engine = template_engine
        self.template_string = template_string

    def render(self, value, values_history=None):
        template = self.template_engine.from_string(self.template_string)

        if not values_history:
            values_history = _generate_empty_historical_data(value)

        return template.render(
            value=value,
            metric_history=values_history,
        )
