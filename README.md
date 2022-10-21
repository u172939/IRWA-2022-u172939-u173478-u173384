# IRWA-2022-u172939-u173478-u173384

First, we should put the path to the inputs directory in the function os.chdir('.../inputs'). This way we change the directory we are in, and we won't need to write the whole path every time we need to access an input file

All necessary imports are in the 'Import libraries' section

## Functions

- clean(text: string): string
- build_terms(text: string): list of strings
- create_mapping(filename, key, value, verbose=True): dictionary. The filename must reffer to a csv

The rest of the code should execute correctly as everything that needed to be defined has been defined above
