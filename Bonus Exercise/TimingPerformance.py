import pandas as pd
import matplotlib.pyplot as plt

def main():
    data = {
        'Number of Processes': [16, 36, 64, 144, 256],
        'Runtime (seconds)': [0.004982, 0.006033, 0.00911, 0.006971, 0.00704]
    }


    df = pd.DataFrame(data)

    print("Runtime Data Table:")
    print(df)
    print("\n")  

    plt.figure(figsize=(10, 6))
    plt.plot(df['Number of Processes'], df['Runtime (seconds)'], marker='o', linestyle='-', color='b')
    plt.title('Runtime vs Number of Processes')
    plt.xlabel('Number of Processes')
    plt.ylabel('Runtime (seconds)')
    plt.grid(True)
    plt.xticks(df['Number of Processes'])  
    plt.tight_layout()
    

    plt.savefig('runtime_plot.png')
    plt.show()

if __name__ == "__main__":
    main()
