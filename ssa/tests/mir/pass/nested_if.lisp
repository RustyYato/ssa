(block 
    (let hello)
    (input hello)
    (if (= hello 0) (set hello 1) (if (= hello 1) (set hello 2) (set hello 3)))
    (print hello))