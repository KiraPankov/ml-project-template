from mlproject.data import make_synthetic, split_data


def test_time_split_has_no_leakage(cfg):
    df = make_synthetic(n_samples=500, seed=0)
    train, test = split_data(df, cfg.data)
    # все объекты теста строго позже всех объектов трейна
    assert train["event_time"].max() < test["event_time"].min()
    assert len(train) + len(test) == len(df)
