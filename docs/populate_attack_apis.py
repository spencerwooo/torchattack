"""
This script dynamically populates the markdown docs for each attack in the `attacks`
directory. Each attack is documented in a separate markdown file with the attack's
name as the filename. The content of each markdown file is dynamically generated
from the docstrings of the attack classes in the `torchattack` module.

Example of the generated docs:

```markdown
# MIFGSM

::: torchattack.MIFGSM
```
"""

import os

import mkdocs_gen_files

import torchattack as ta

for attack_name in ta.SUPPORTED_ATTACKS:
    filename = os.path.join('attacks', f'{attack_name.lower()}.md')
    with mkdocs_gen_files.open(filename, 'w') as f:
        f.write(f'# {attack_name}\n\n::: torchattack.{attack_name}\n')
    mkdocs_gen_files.set_edit_path(filename, 'docs/populate_attack_apis.py')
