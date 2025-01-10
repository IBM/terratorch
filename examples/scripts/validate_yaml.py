#!/usr/bin/env python3
import argparse
import os

import yamale

"""
usage:
python3 ../scripts/validate_yaml.py  <file.yaml>
"""

homedir = os.path.expanduser('~')
schema_path = homedir+"/terratorch/examples/confs/"
cwd = os.getcwd()

# Get filename of yaml file to validate
if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Validate configuration file (YAML)')
    parser.add_argument('file',
                        action='store',
                        metavar='INPUT_FILE',
                        type=str,
                        help='Yaml file containing configuration to be validated')
    arg = parser.parse_args()

# Create the Schema object
schemafile = 'yaml_schema.yaml'
schema = yamale.make_schema(schema_path+schemafile)

# Create a Data object
data = yamale.make_data(cwd+'/'+arg.file)

try:
    yamale.validate(schema, data)
    print('Validation success! 👍')
except ValueError as e:
    print('Validation failed!\n')
    for result in e.results:
        print("Error validating data '%s' with '%s'\n\t" % (result.data, result.schema))
        for error in result.errors:
            print('\t%s' % error)
    exit(1)


