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

for attack in ta.SUPPORTED_ATTACKS:
    filename = os.path.join('attacks', f'{attack.lower()}.md')
    with mkdocs_gen_files.open(filename, 'w') as f:
        if attack in ta.GENERATIVE_ATTACKS:
            # For generative attacks, we need to document
            # both the attack and its weights enum as well
            attack_mod = getattr(ta, attack.lower())
            weights_enum = getattr(attack_mod, f'{attack}Weights')
            weights_doc = [f'- `{w}`\n' for w in weights_enum.__members__]
            f.write(
                f'# {attack}\n\n'
                f'::: torchattack.{attack}\n'
                f'::: torchattack.{attack.lower()}.{attack}Weights\n\n'
                f'Available weights:\n\n'
                f'{"".join(weights_doc)}\n'
            )
        else:
            f.write(f'# {attack}\n\n::: torchattack.{attack}\n')
    mkdocs_gen_files.set_edit_path(filename, 'docs/populate_attack_apis.py')
