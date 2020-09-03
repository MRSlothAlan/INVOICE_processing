"""
Define all the rules for document extraction in here!
Build up a database of rules for checking
"""

"""
Notes on different names:

total amount due
total payable hkd
grand total
billing customer
attn:
special offer
quotation (header)
tax invoice number


"""

"""
=== CLIENT ===
client:
To:
"""

"""
=== HEADERS ===

"""

"""
Supplier

payable to
kindly arrage payment to
"""

"""
Notes: 

I can include all possible entries behind the colon
"""

# add a weighting to replace the need of word model i.e. no need feature vectors
# since i don't have time and dataset to generate a weighted token
# scheme: from 0 to 1, what is the score?

"""
TO-DO
"""
POSSIBLE_HEADER_WORDS = {
    "item" : 0.2,
    "model" : 0.3,
    "description" : 0.9,
    "descriptions" : 0.9,
    "qty" : 0.6,
    "quantity" : 0.7,
    "unit" : 0.4,
    "unit price" : 0.6,
    "amount" : 0.4,
    "gst" : 0.2,
    "customer" : 0.2,
    "product" : 0.5,
    "number" : 0.1,
    "bill" : 0.3,
    "period" : 0.2,
    "total" : 0.4,
    "details" : 0.8,
    "detail" : 0.9,
    "particulars" : 0.7,
    "po.no.": 0.6,
    "net amount": 0.5,
    "net" : 0.6,
    "po" : 0.8
}
