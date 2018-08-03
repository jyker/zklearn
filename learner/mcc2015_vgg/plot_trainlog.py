import csv
import matplotlib.pyplot as plt

def main(training_log):
    with open(training_log) as fin:
        reader = csv.reader(fin)
        next(reader, None)  
        accuracies = []
        loss = []
        val_accuracies = []
        val_loss = []
        benchmark_acc = []
        benchmark_loss = []  
        for _, acc, tr_l, val_acc, val_l in reader:
            accuracies.append(float(acc))
            loss.append(float(tr_l))
            val_accuracies.append(float(val_acc))
            val_loss.append(float(val_l))
            benchmark_acc.append(0.80)
            benchmark_loss.append(0.30)
        # loss 
        plt.plot(loss)
        plt.plot(val_loss)
        plt.plot(benchmark_loss)
        # plt.plot(benchmark)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid', '0.30'], loc='upper right')
        plt.show()
        # acc
        plt.plot(accuracies)
        plt.plot(val_accuracies)
        plt.plot(benchmark_acc)
        # plt.plot(benchmark)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid', '0.80'], loc='lower right')
        plt.show()


if __name__ == '__main__':
    training_log = 'logs/asm_kong_1_6400[0].log'
    main(training_log)

