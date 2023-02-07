from create_multiple_model_sup_untrans_bin import train
from generate_input import generate
from test_sup_multiple import test
from transform_graph import transform


def main():
    # generate()
    # transform()
    # train()
    for neighbors_positive in [True, False]:
        test(neighbors_positive)


if __name__ == '__main__':
    main()
