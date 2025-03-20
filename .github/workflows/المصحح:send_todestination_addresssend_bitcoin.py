send_todestination_addresssend_bitcoin.py
from bitcoinlib.wallets import Wallet

# Open the wallet
wallet = Wallet('MyWallet')  # Ensure that the wallet "MyWallet" has been created and contains the private key

# Destination address
destination_address = 'bc1q59pelcp7z9k208jzh0q6kdxz4md98e8ws2mv2k'  # Replace with the actual destination address

# Amount in Bitcoin (BTC)
amount_btc = 1.0

# Create and send the transaction
tx = wallet.send_to(destination_address, amount_btc)

# Print transaction details
print("Transaction ID:", tx.txid)
print("Transaction details:", tx.info())
