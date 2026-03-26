from genesis.main import GenesisMind

def test():
    print("--- Initializing Genesis ---")
    mind = GenesisMind()
    
    print("\\n--- Teaching Concept 1 ---")
    res1 = mind.teach_concept("apple", use_camera=False)
    print(res1)
    
    print("\\n--- Teaching Concept 2 ---")
    res2 = mind.teach_concept("banana", use_camera=False)
    print(res2)
    
    print("\\n--- Teaching Concept 3 ---")
    res3 = mind.teach_concept("cherry", use_camera=False)
    print(res3)
    
    print("\\n--- Triggering Sleep Compilation (Consolidation) ---")
    res_sleep = mind.trigger_sleep()
    print(res_sleep)
    
    print("\\n--- Recall test ---")
    res_recall = mind.recall_concept("apple")
    print("Recall apple:", res_recall)
    
    print("\\nAll ML passes successful!")

if __name__ == "__main__":
    test()
