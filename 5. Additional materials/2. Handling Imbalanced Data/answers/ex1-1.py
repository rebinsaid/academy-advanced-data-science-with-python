
from imblearn.under_sampling import TomekLinks, NearMiss, OneSidedSelection

tl = TomekLinks()
nm = NearMiss(version = 3)
oss = OneSidedSelection()