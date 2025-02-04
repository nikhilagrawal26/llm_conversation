#!/usr/bin/env python

"""
Generate JSON schema for the configuration file using Pydantic.
"""

import json

from src.llm_conversation.config import Config


def main():
    # Generate current schema
    current_schema = Config.model_json_schema()

    # Output schema to stdout
    print(json.dumps(current_schema, indent=4))


if __name__ == "__main__":
    main()
