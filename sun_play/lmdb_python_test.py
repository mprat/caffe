import lmdb

env = lmdb.open('sun_testing_01_lmdb_0_start/')

with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        print ((key, value))
        raw_input()
