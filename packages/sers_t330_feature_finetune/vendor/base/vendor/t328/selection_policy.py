"""Protect both all-correct spectra and individual grades within each mixture."""
DRUGS = ('DEL', 'CHL', 'TEB')
VARIANTS = ('adam_control', 'adamw', 'layerwise', 'lora', 'adapter')


def rank(metrics):
    return (metrics['full_profile_correct'], metrics['present_correct'],
            metrics['ternary_all_grade_correct'], -metrics['present_mae'])


def acceptance(metrics, anchor):
    rules = {}
    for key in ('classification_exact_correct', 'full_profile_correct',
                'present_correct', 'ternary_all_grade_correct'):
        rules[key] = metrics[key] >= anchor[key]
    rules['classification_macro_f1'] = metrics['classification_macro_f1'] + 1e-12 >= anchor['classification_macro_f1']
    for drug in DRUGS:
        rules['overall_grade_' + drug] = metrics['per_drug'][drug]['correct'] >= anchor['per_drug'][drug]['correct']
    for count in ('1', '2', '3'):
        current = metrics['by_mixture'][count]
        reference = anchor['by_mixture'][count]
        for key in ('classification_exact_correct', 'full_profile_correct', 'present_correct'):
            rules[f'mixture_{count}_{key}'] = current[key] >= reference[key]
        for drug in DRUGS:
            rules[f'mixture_{count}_grade_{drug}'] = current['per_drug'][drug]['correct'] >= reference['per_drug'][drug]['correct']
    rules['strict_improvement'] = any(metrics[key] > anchor[key] for key in (
        'full_profile_correct', 'present_correct', 'ternary_all_grade_correct'))
    return dict(accepted=all(rules.values()), rules=rules,
                failed_rules=[key for key, value in rules.items() if not value])
