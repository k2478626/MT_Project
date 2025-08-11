from sklearn.model_selection import train_test_split
import numpy as np
import os

def split_and_save_real_data(real_data, test_size=0.2, random_state=42, output_dir=r"C:\Users\awalk\OneDrive\Desktop\Kings College London\Individual Project\test\stEVE_training\training_scripts"):
    """
    Splits real_data into train/test sets and saves them to disk.

    Parameters:
        real_data (dict): Dict with keys 'observations', 'actions', 'next_observations', 'terminals'
        test_size (float): Fraction of the data to reserve for testing
        random_state (int): Random seed
        output_dir (str): Directory where .npy files will be saved

    Returns:
        train_data (dict), test_data (dict)
    """
    combined = list(zip(
        real_data["observations"],
        real_data["actions"],
        real_data["next_observations"],
        real_data["terminals"]
    ))

    train_combined, test_combined = train_test_split(
        combined, test_size=test_size, random_state=random_state
    )

    def unpack(data):
        return {
            "observations":      np.array([x[0] for x in data]),
            "actions":           np.array([x[1] for x in data]),
            "next_observations": np.array([x[2] for x in data]),
            "terminals":         np.array([x[3] for x in data])
        }

    train_data = unpack(train_combined)
    test_data = unpack(test_combined)

    # Save to disk
    train_path = os.path.join(output_dir, "real_data_train.npy")
    test_path = os.path.join(output_dir, "real_data_test.npy")
    np.save(train_path, train_data)
    np.save(test_path, test_data)

    print(f"[✔] Train data saved to {train_path}")
    print(f"[✔] Test data saved to {test_path}")

    return train_data, test_data

real_data = np.load(r"C:\Users\awalk\OneDrive\Desktop\Kings College London\Individual Project\test\stEVE_training\training_scripts\combined_cleaned_data.npy", allow_pickle=True).item()

train_data, test_data = split_and_save_real_data(real_data, test_size=0.2)
