import os
import time
import requests
import statistics
from datetime import datetime
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from pybit.unified_trading import WebSocket

# Load environment variables
load_dotenv('config/.env')

api_key = os.getenv('BYBIT_API_KEY')
secret_key = os.getenv('BYBIT_SECRET_KEY')
SYMBOL = os.getenv('SYMBOL', '1000PEPEUSDT')

# ============================================
# LATENCY TEST FUNCTIONS
# ============================================

def test_http_latency(session, endpoint_name, test_func, iterations=5):
    """Test latency of a specific HTTP endpoint"""
    latencies = []
    errors = 0
    
    print(f"\n📡 Testing {endpoint_name}...")
    
    for i in range(iterations):
        try:
            start = time.perf_counter()
            result = test_func(session)
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            
            print(f"  Iteration {i+1}: {latency_ms:.2f} ms")
            
        except Exception as e:
            errors += 1
            print(f"  Iteration {i+1}: ❌ Error - {str(e)[:60]}...")
    
    if latencies:
        avg_latency = statistics.mean(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0
        
        print(f"\n  📊 {endpoint_name} Results ({len(latencies)} successful):")
        print(f"     Average: {avg_latency:.2f} ms")
        print(f"     Min:     {min_latency:.2f} ms")
        print(f"     Max:     {max_latency:.2f} ms")
        print(f"     Std Dev: {std_dev:.2f} ms")
        print(f"     Errors:  {errors}")
        
        return {
            'avg': avg_latency,
            'min': min_latency,
            'max': max_latency,
            'std_dev': std_dev,
            'errors': errors,
            'iterations': len(latencies)
        }
    else:
        print(f"  ❌ All {iterations} iterations failed!")
        return None

def test_websocket_latency():
    """Test WebSocket connection latency"""
    print("\n🔌 Testing WebSocket connection...")
    
    try:
        start = time.perf_counter()
        
        ws = WebSocket(
            testnet=False,
            channel_type="linear",
            api_key=api_key,
            api_secret=secret_key,
        )
        
        connect_time = (time.perf_counter() - start) * 1000
        print(f"  ✅ WebSocket connected in: {connect_time:.2f} ms")
        
        ws.exit()
        return connect_time
        
    except Exception as e:
        print(f"  ❌ WebSocket error: {e}")
        return None

# ============================================
# HTTP ENDPOINT TEST FUNCTIONS
# ============================================

def test_get_balance(session):
    """Test wallet balance endpoint"""
    response = session.get_wallet_balance(
        accountType='UNIFIED',
        coin='USDT'
    )
    return response

def test_get_orderbook(session):
    """Test order book endpoint"""
    response = session.get_orderbook(
        category="linear",
        symbol=SYMBOL,
        limit=10
    )
    return response

def test_get_ticker(session):
    """Test ticker endpoint"""
    response = session.get_tickers(
        category="linear",
        symbol=SYMBOL
    )
    return response

# ============================================
# HOST INFO
# ============================================

def get_host_info():
    """Get information about current host"""
    try:
        ip_response = requests.get('https://api.ipify.org', timeout=5)
        public_ip = ip_response.text
        
        location_response = requests.get(f'http://ip-api.com/json/{public_ip}', timeout=5)
        location_data = location_response.json()
        
        return {
            'ip': public_ip,
            'city': location_data.get('city', 'Unknown'),
            'country': location_data.get('country', 'Unknown'),
            'isp': location_data.get('isp', 'Unknown'),
            'timezone': location_data.get('timezone', 'Unknown')
        }
    except:
        return {
            'ip': 'Unknown',
            'city': 'Unknown',
            'country': 'Unknown',
            'isp': 'Unknown',
            'timezone': 'Unknown'
        }

# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("🚀 BYBIT API LATENCY TEST SUITE")
    print("=" * 60)
    
    # Host info
    print("\n📍 HOST INFORMATION")
    print("-" * 40)
    host_info = get_host_info()
    print(f"  IP Address:    {host_info['ip']}")
    print(f"  Location:      {host_info['city']}, {host_info['country']}")
    print(f"  ISP:           {host_info['isp']}")
    print(f"  Timezone:      {host_info['timezone']}")
    print(f"  Test Time:     {datetime.utcnow().isoformat()}")
    print(f"  Symbol:        {SYMBOL}")
    
    # API keys
    print("\n🔑 API KEY STATUS")
    print("-" * 40)
    if not api_key or not secret_key:
        print("  ❌ API keys not found in .env file!")
        return
    
    print(f"  API Key:       {api_key[:10]}...{api_key[-4:]}")
    print(f"  Secret Key:    {secret_key[:10]}...{secret_key[-4:]}")
    print("  ✅ Keys loaded successfully")
    
    # Session
    print("\n🔗 CONNECTING TO BYBIT")
    print("-" * 40)
    
    try:
        session = HTTP(
            testnet=False,
            api_key=api_key,
            api_secret=secret_key,
        )
        print("  ✅ HTTP session created")
    except Exception as e:
        print(f"  ❌ Failed to create session: {e}")
        return
    
    # Tests
    print("\n⏱️  LATENCY TESTS")
    print("=" * 60)
    
    results = {}
    
    results['balance'] = test_http_latency(session, "GET /wallet-balance", test_get_balance)
    results['orderbook'] = test_http_latency(session, "GET /orderbook", test_get_orderbook)
    results['ticker'] = test_http_latency(session, "GET /ticker", test_get_ticker)
    
    ws_latency = test_websocket_latency()
    results['websocket'] = {'avg': ws_latency} if ws_latency else None
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 LATENCY SUMMARY")
    print("=" * 60)
    
    summary_data = {}
    for name, result in results.items():
        if result and 'avg' in result and result['avg']:
            latency = result['avg']
            print(f"  {name:15s}: {latency:>8.2f} ms")
            summary_data[name] = latency
    
    # Rating
    print("\n🏆 PERFORMANCE RATING")
    print("-" * 40)
    
    if summary_data:
        avg_all = statistics.mean(summary_data.values())
        
        if avg_all < 50:
            rating = "⭐ EXCELLENT - Suitable for HFT"
        elif avg_all < 100:
            rating = "⭐⭐ GOOD - Suitable for scalping"
        elif avg_all < 200:
            rating = "⭐⭐⭐ AVERAGE - Suitable for swing trading"
        else:
            rating = "⭐⭐⭐⭐ POOR - Consider moving to cloud VPS"
        
        print(f"  Overall Average: {avg_all:.2f} ms")
        print(f"  Rating:          {rating}")
        
        if avg_all > 150:
            print("\n  💡 Recommendation:")
            print("     Your latency is high (>150ms). Consider:")
            print("     1. Using a VPS in the same region as Bybit (Tokyo, Singapore)")
            print("     2. Testing with a cloud provider (AWS, DigitalOcean)")
    
    # Save
    print("\n💾 SAVING RESULTS")
    print("-" * 40)
    
    import csv
    from pathlib import Path
    
    repo_root = Path(__file__).resolve().parents[1]
    results_file = repo_root / 'data' / 'latency_results.csv'
    results_file.parent.mkdir(exist_ok=True)
    
    file_exists = results_file.exists()
    
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'host_ip', 'host_location', 'balance_latency', 
                           'orderbook_latency', 'ticker_latency', 'websocket_latency', 'overall_avg'])
        
        row = [
            datetime.utcnow().isoformat(),
            host_info['ip'],
            f"{host_info['city']}, {host_info['country']}",
            summary_data.get('balance', 0),
            summary_data.get('orderbook', 0),
            summary_data.get('ticker', 0),
            summary_data.get('websocket', 0),
            statistics.mean(summary_data.values()) if summary_data else 0
        ]
        writer.writerow(row)
    
    print(f"  ✅ Results saved to: {results_file}")
    
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()