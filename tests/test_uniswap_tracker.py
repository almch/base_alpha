from blockchain.interactions.uniswap_tracker import UniswapTracker

def test_uniswap_tracker():
    tracker = UniswapTracker()
    swaps = tracker.get_latest_swaps()
    tracker.print_swaps(swaps)

if __name__ == "__main__":
    test_uniswap_tracker() 