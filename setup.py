import setuptools

setuptools.setup(
    name="MTGpred",
    use_scm_version=True,
    author="Javier Jimenez",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    setup_requires=["setuptools_scm"],
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "transformers",
        "deepspeed",
        "pymongo",
        "wandb",
        "typer",
        "sentencepiece",
        "black",
        #"mpi4py", Install with conda
    ],
)
