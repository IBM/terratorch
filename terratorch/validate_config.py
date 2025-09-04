import json
import yaml
import argparse
from jsonschema import validate, ValidationError, SchemaError

def main():
    parser = argparse.ArgumentParser(description="Validate a YAML file against a JSON Schema.")
    parser.add_argument("schema", help="Path to the JSON Schema")
    parser.add_argument("yaml", help="Path to the YAML")
    args = parser.parse_args()

    try:
        with open(args.schema, "r", encoding="utf-8") as file:
            schema = json.load(file)
        with open(args.yaml, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)
        validate(instance=yaml_data, schema=schema)
        print('OK')
    except ValidationError as e:
        print(f"Validation error: {e.message}")
    except SchemaError as e:
        print(f"Schema error: {e.message}")
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
    except json.JSONDecodeError:
        print("Invalid JSON in schema file.")
    except yaml.YAMLError:
        print("Invalid YAML in data file.")

if __name__ == "__main__":
    main()
