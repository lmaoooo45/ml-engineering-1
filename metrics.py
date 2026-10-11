def classification_metrics(expected, predicted, labels):
    if not expected or len(expected) != len(predicted):
        raise ValueError("Нужны непустые списки одинаковой длины")
    if any(x not in labels for x in expected + predicted):
        raise ValueError("Неизвестная метка класса")
    matrix = [[sum(a == row and b == col for a, b in zip(expected, predicted))
               for col in labels] for row in labels]
    f1 = []
    for i in range(len(labels)):
        tp = matrix[i][i]
        fp = sum(row[i] for row in matrix) - tp
        fn = sum(matrix[i]) - tp
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {"n": len(expected), "accuracy": sum(a == b for a, b in zip(expected, predicted)) / len(expected),
            "macro_f1": sum(f1) / len(labels), "labels": labels, "confusion_matrix": matrix}
