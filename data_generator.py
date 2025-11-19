import pandas as pd
import random
import string
from faker import Faker

def generate_data(n_unique=200, n_duplicates=40, seed=42):
    """
    Generates a synthetic dataset of PII (Personally Identifiable Information) records
    for testing probabilistic record matching.

    Reasoning for Synthetic Data:
    -----------------------------
    1.  **Ground Truth**: In real-world record linkage, we rarely know for sure which records match.
        By generating data, we control the `original_id`, giving us perfect ground truth labels
        to train and evaluate our model.
    2.  **Controlled Variations**: We can explicitly control the types and frequency of errors
        (typos, missing data) to stress-test the matching logic.

    Assumptions:
    ------------
    1.  **Randomness**: We assume errors (typos) and missing data occur randomly. In reality,
        errors might be systematic (e.g., specific OCR confusion pairs like 'l' vs '1').
    2.  **Independence**: We assume errors in one field (e.g., Name) are independent of errors
        in another (e.g., Address).
    3.  **Distribution**: We assume a roughly 80/20 split of unique vs. duplicate records is
        sufficient for a pilot.

    Parameters:
    -----------
    n_unique : int
        Number of unique identities to generate.
    n_duplicates : int
        Number of duplicate records to generate by perturbing the unique ones.
    seed : int
        Random seed for reproducibility.
    """
    fake = Faker()
    Faker.seed(seed)
    random.seed(seed)
    
    def perturb(text):
        """
        Introduces random character-level errors to simulate typos or OCR mistakes.
        
        Types of errors:
        - Deletion: Missing a character.
        - Insertion: Extra character added.
        - Replacement: Wrong character typed.
        - Swap: Adjacent characters transposed.
        
        Assumption:
        - 30% chance of error per call is high enough to be challenging but low enough
          to preserve similarity.
        """
        if not text or random.random() > 0.3: return text
        chars = list(text)
        op = random.choice(['del', 'ins', 'rep', 'swap'])
        idx = random.randint(0, len(chars)-1)
        if op == 'del': 
            del chars[idx]
        elif op == 'ins': 
            chars.insert(idx, random.choice(string.ascii_letters))
        elif op == 'rep': 
            chars[idx] = random.choice(string.ascii_letters)
        elif op == 'swap' and len(chars) > 1: 
            idx2 = idx + 1 if idx < len(chars) - 1 else idx - 1
            chars[idx], chars[idx2] = chars[idx2], chars[idx]
        return "".join(chars)

    data = []
    
    # --- Step 1: Generate Unique Records ---
    # We create 'n_unique' distinct identities. These represent the "clean" database.
    for _ in range(n_unique):
        uid = fake.uuid4()
        record = {
            'id': uid, 
            'original_id': uid, # Ground truth identifier
            'name': fake.name(), 
            'address': fake.address().replace('\n', ', '),
            'city': fake.city(), 
            'date_of_birth': str(fake.date_of_birth())
        }
        
        # Introduce Missing Data (None)
        # Reasoning: Real data is rarely complete. We simulate this by randomly dropping fields.
        # Assumption: 10% missing rate per field is a reasonable baseline for "dirty" data.
        for field in ['address', 'city', 'date_of_birth']:
            if random.random() < 0.1:
                record[field] = None
        data.append(record)
    
    # --- Step 2: Generate Duplicates ---
    # We create 'n_duplicates' records that are variations of the unique ones.
    # These represent the "dirty" incoming records we want to match.
    for _ in range(n_duplicates):
        # Pick a random existing record to duplicate
        orig = random.choice(data[:n_unique])
        dup = orig.copy()
        
        # Perturb fields to simulate data entry errors
        # We assign a NEW 'id' but keep the SAME 'original_id' to track the match.
        dup.update({
            'id': fake.uuid4(), 
            'name': perturb(orig['name']), 
            # Only perturb address if it exists (handle None case)
            'address': perturb(orig['address']) if orig['address'] else None
        })
        
        # Independent chance of missing data in the duplicate
        # A record might have an address in the master list but be missing it in the new list.
        for field in ['address', 'city', 'date_of_birth']:
            if random.random() < 0.1:
                dup[field] = None
                
        data.append(dup)
        
    return pd.DataFrame(data)
