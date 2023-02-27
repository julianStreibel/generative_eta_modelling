if __name__ == "__main__":
    # %%
    import spn
    from utils.read_data import read_data
    from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
    from spn.algorithms.LearningWrappers import learn_parametric
    from spn.structure.Base import Context

    # %%
    train_data = read_data()
    train_data = train_data.dropna()
    train_data = train_data.to_numpy()

    # %%
    context = Context(
        parametric_types=[
            Gaussian,
            Gaussian,
            Gaussian,
            Gaussian,
            Gaussian,
            Gaussian,
            Gaussian,
            Categorical,
            Categorical,
            Categorical,
            Categorical,
            Categorical,
            Categorical,
            Categorical,
            Categorical,
            Gaussian,
            Gaussian,
            Gaussian,
            ]
        ).add_domains(train_data)

    # %%
    spn_classification = learn_parametric(
        train_data,
        context,
        min_instances_slice=1000
    )

    # %%
