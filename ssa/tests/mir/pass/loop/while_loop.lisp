(block
    (let x)
    (set x 100)
    (loop 
        (let y)
        (set y 0)
        (if x
            (block
                (print x)
                (set x (- x 1)))
            (break ))))