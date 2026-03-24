def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    TP = 0
    num_preds = len(y_pred)

    for i in range(num_preds):
        if y_pred[i] == y_true[i]:
            TP += 1

        F1_m =  TP / num_preds

    return F1_m