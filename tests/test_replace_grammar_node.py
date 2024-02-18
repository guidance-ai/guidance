from guidance._grammar import Join, Byte, Placeholder

def test_simple_replacement():
    target = Placeholder()
    replacement = Byte(b'c')
    grammar = Join(['a', 'b', target])
    target.set(replacement)
    assert grammar.values == [Byte(b'a'), Byte(b'b'), Byte(b'c')]

def test_multiple_replacement():
    target1 = Placeholder()
    replacement1 = Byte(b'b')
    target2 = Placeholder()
    replacement2 = Byte(b'c')
    grammar = Join(['a', target1, target2])
    target1.set(replacement1)
    target2.set(replacement2)
    assert grammar.values == [Byte(b'a'), Byte(b'b'), Byte(b'c')]

def test_chained_replacement():
    target1 = Placeholder()
    replacement1 = Byte(b'c')
    grammar1 = Join(['b', target1])

    target2 = Placeholder()
    replacement2 = grammar1
    grammar2 = Join(['a', target2])

    target1.set(replacement1)
    target2.set(replacement2)

    assert grammar1.values == [Byte(b'b'), Byte(b'c')]
    assert grammar2.values == [Byte(b'a'), grammar1]

def test_recursive_replacement():
    target = Placeholder()
    grammar = Join(['a', 'b', target])
    target.set(grammar)
    assert grammar.values == [Byte(b'a'), Byte(b'b'), grammar]

def test_chained_recursive_replacement():
    target = Placeholder()
    grammar1 = Join(['b', target])
    grammar2 = Join(['a', target])
    
    target.set(grammar1)

    assert grammar2.values == [Byte(b'a'), grammar1]
    assert grammar1.values == [Byte(b'b'), grammar1]