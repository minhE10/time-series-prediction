def first_value(choices):
    if not choices:
        return None
    first = choices[0]
    return first[1] if isinstance(first, tuple) else first


def target_choices(registry, dataset_name):
    if not dataset_name:
        return []
    return registry.datasets[dataset_name].target_cols
