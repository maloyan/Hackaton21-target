def compute_accuracy(target, pred):

    targets_ammount = [0 for i in range(5)]
    correct_preds = [0 for i in range(5)]

    for i, j in zip(target, pred):
        targets_ammount[i] += 1
        if i == j:
            correct_preds[i] += 1

    avg_acc = sum(correct_preds) / sum(targets_ammount)

    per_target_acc = [
        correct_preds[i] / targets_ammount[i] for i in range(5) if targets_ammount[i]
    ]

    weights = [0.4, 0.3, 0.09, 0.12, 0.06]
    absolute_score = sum(per_target_acc[i] / weights[i] for i in range(5))

    print(
        "avg_acc {}, per_class_acc human:{} target_human:{} target_laser:{} target_gun:{} target_tank:{}".format(
            avg_acc, *per_target_acc
        )
    )
    print("score {}".format(absolute_score))

    return avg_acc, per_target_acc
