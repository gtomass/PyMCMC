from setuptools import setup, find_packages

# Note: La plupart des réglages sont maintenant dans pyproject.toml
# On garde setup() vide ou minimal pour la compatibilité avec l'installation éditable (-e)
setup(
    packages=find_packages(),
)
