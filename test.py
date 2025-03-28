import cProfile

def test_function():
    print("Testing 1, 2, 3...")
    for i in range(10000):
        pass

cProfile.run('test_function()')
