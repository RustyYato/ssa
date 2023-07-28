(block
    (let x)
    (set x 100)
    (loop 
        (let y)
        (if x (block
            (print x)
            (set x (+ x 1))
            (continue )))))