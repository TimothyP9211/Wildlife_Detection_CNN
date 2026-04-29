def match_bboxes(IoU, threshold=0.5):
    matches = []
    matched_pred = set()
    matched_target = set()

    candidates = []

    for i in range(IoU.shape[0]):
        for j in range(IoU.shape[1]):
            candidates.append((IoU[i, j].item(), i, j))

    candidates.sort(reverse=True)

    for iou, pred_idx, target_idx in candidates:
        if iou < threshold:
            continue
        if (pred_idx in matched_pred) or (target_idx in matched_target):
            continue

        matched_pred.add(pred_idx)
        matched_target.add(target_idx)
        matches.append((pred_idx, target_idx))
    
    return matches

