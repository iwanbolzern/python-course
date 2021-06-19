
class Example:
    class_attr = 0

    def __init__(self, instance_attr: int) -> None:
        self.instance_attr = instance_attr


if __name__ == '__main__':
    ex1 = Example(1)
    ex2 = Example(2)

    print("Example.class_attr =", Example.class_attr)
    print("ex1.class_attr =", ex1.class_attr)
    print("ex2.class_attr =", ex2.class_attr)
    print("ex1.instance_attr =", ex1.instance_attr)
    print("ex2.instance_attr =", ex2.instance_attr)

    # Example.instance_attr  # AttributeError

    print("\nChanging class_attr from the class:")
    Example.class_attr = 5
    print("Example.class_attr =", Example.class_attr)
    print("ex1.class_attr =", ex1.class_attr)
    print("ex2.class_attr =", ex2.class_attr)

    print("\nChanging class_attr from ex1:")
    ex1.class_attr = 10
    print("ex1.class_attr =", ex1.class_attr)
    print("ex2.class_attr =", ex2.class_attr)
    print("Example.class_attr =", Example.class_attr)

    print("ex1 inspection:", ex1.__dict__)
    print("ex2 inspection:", ex2.__dict__)
