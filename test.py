import scipy 


def main():
    test = scipy.io.loadmat('C:/Users/Daniil/armband/IEEE-NER-2023-EffiE/Ninapro_DB5/S1/S1_E1_A1.mat')

    print(test['emg'].shape)

if __name__ == '__main__':
    main()