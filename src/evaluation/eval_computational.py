import json
import os
from pathlib import Path
from collections import defaultdict
import csv
from statistics import mean, stdev
from datetime import datetime


def load_log_files(logs_dir):
    """Load all JSON log files from the directory."""
    logs = []
    log_files = sorted(Path(logs_dir).glob("*.json"))
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
                logs.append(log_data)
        except Exception as e:
            print(f"Error loading {log_file}: {e}")
    
    return logs


def calculate_stats(values):
    """Calculate mean, stdev, min, max for a list of values."""
    if not values:
        return {}
    
    stats = {
        "mean": round(mean(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "count": len(values)
    }
    
    if len(values) > 1:
        stats["stdev"] = round(stdev(values), 2)
    
    return stats


def aggregate_computational_logs(logs):
    """Aggregate logs by model and calculate statistics."""
    model_stats = defaultdict(lambda: {
        "times": [],
        "cpu_percents": [],
        "ram_peaks": [],
        "gpu_utils": [],
        "gpu_mem_used": [],
        "gpu_mem_percents": [],
        "gpu_temps": [],
        "success_count": 0,
        "total_count": 0,
        "timestamps": []
    })
    
    # Aggregate data
    for log in logs:
        model = log["metadata"]["model"]
        timestamp = log["metadata"].get("timestamp", "")
        model_stats[model]["timestamps"].append(timestamp)
        
        for record in log.get("records", []):
            model_stats[model]["total_count"] += 1
            
            if record.get("status") == "SUCCESS":
                model_stats[model]["success_count"] += 1
                model_stats[model]["times"].append(record.get("time_sec", 0))
                model_stats[model]["cpu_percents"].append(record.get("cpu_percent", 0))
                model_stats[model]["ram_peaks"].append(record.get("ram_peak_mb", 0))
                
                gpu = record.get("gpu", {})
                if gpu:
                    model_stats[model]["gpu_utils"].append(gpu.get("util", 0))
                    model_stats[model]["gpu_mem_used"].append(gpu.get("mem_used", 0))
                    model_stats[model]["gpu_mem_percents"].append(gpu.get("mem_percent", 0))
                    model_stats[model]["gpu_temps"].append(gpu.get("temp_peak", 0))
    
    return model_stats


def generate_summary(model_stats):
    """Generate summary statistics for each model."""
    summary = {}
    
    for model, stats in model_stats.items():
        summary[model] = {
            "timestamp": stats["timestamps"][0] if stats["timestamps"] else "",
            "total_records": stats["total_count"],
            "successful_records": stats["success_count"],
            "success_rate": round((stats["success_count"] / stats["total_count"] * 100) if stats["total_count"] > 0 else 0, 2),
            "execution_time": calculate_stats(stats["times"]),
            "cpu_usage": calculate_stats(stats["cpu_percents"]),
            "ram_peak": calculate_stats(stats["ram_peaks"]),
            "gpu_utilization": calculate_stats(stats["gpu_utils"]),
            "gpu_memory_used": calculate_stats(stats["gpu_mem_used"]),
            "gpu_memory_percent": calculate_stats(stats["gpu_mem_percents"]),
            "gpu_temp_peak": calculate_stats(stats["gpu_temps"]),
        }
    
    return summary


def save_to_json(summary, output_path):
    """Save summary to JSON file."""
    os.makedirs(output_path, exist_ok=True)
    
    json_file = os.path.join(output_path, "computational_summary.json")
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ JSON report saved to {json_file}")
    return json_file


def save_to_csv(summary, output_path):
    """Save summary to CSV file."""
    os.makedirs(output_path, exist_ok=True)
    
    csv_file = os.path.join(output_path, "computational_summary.csv")
    
    # Flatten the summary for CSV
    rows = []
    for model, metrics in summary.items():
        row = {
            "Model": model,
            "Timestamp": metrics["timestamp"],
            "Total Records": metrics["total_records"],
            "Successful Records": metrics["successful_records"],
            "Success Rate (%)": metrics["success_rate"],
            "Execution Time Mean (s)": metrics["execution_time"].get("mean", "N/A"),
            "Execution Time Stdev (s)": metrics["execution_time"].get("stdev", "N/A"),
            "Execution Time Min (s)": metrics["execution_time"].get("min", "N/A"),
            "Execution Time Max (s)": metrics["execution_time"].get("max", "N/A"),
            "CPU Usage Mean (%)": metrics["cpu_usage"].get("mean", "N/A"),
            "CPU Usage Stdev (%)": metrics["cpu_usage"].get("stdev", "N/A"),
            "CPU Usage Min (%)": metrics["cpu_usage"].get("min", "N/A"),
            "CPU Usage Max (%)": metrics["cpu_usage"].get("max", "N/A"),
            "RAM Peak Mean (MB)": metrics["ram_peak"].get("mean", "N/A"),
            "RAM Peak Stdev (MB)": metrics["ram_peak"].get("stdev", "N/A"),
            "RAM Peak Min (MB)": metrics["ram_peak"].get("min", "N/A"),
            "RAM Peak Max (MB)": metrics["ram_peak"].get("max", "N/A"),
            "GPU Util Mean (%)": metrics["gpu_utilization"].get("mean", "N/A"),
            "GPU Util Stdev (%)": metrics["gpu_utilization"].get("stdev", "N/A"),
            "GPU Util Min (%)": metrics["gpu_utilization"].get("min", "N/A"),
            "GPU Util Max (%)": metrics["gpu_utilization"].get("max", "N/A"),
            "GPU Mem Used Mean (MB)": metrics["gpu_memory_used"].get("mean", "N/A"),
            "GPU Mem Used Stdev (MB)": metrics["gpu_memory_used"].get("stdev", "N/A"),
            "GPU Mem Used Min (MB)": metrics["gpu_memory_used"].get("min", "N/A"),
            "GPU Mem Used Max (MB)": metrics["gpu_memory_used"].get("max", "N/A"),
            "GPU Mem Percent Mean (%)": metrics["gpu_memory_percent"].get("mean", "N/A"),
            "GPU Mem Percent Stdev (%)": metrics["gpu_memory_percent"].get("stdev", "N/A"),
            "GPU Mem Percent Min (%)": metrics["gpu_memory_percent"].get("min", "N/A"),
            "GPU Mem Percent Max (%)": metrics["gpu_memory_percent"].get("max", "N/A"),
            "GPU Temp Peak Mean (°C)": metrics["gpu_temp_peak"].get("mean", "N/A"),
            "GPU Temp Peak Stdev (°C)": metrics["gpu_temp_peak"].get("stdev", "N/A"),
            "GPU Temp Peak Min (°C)": metrics["gpu_temp_peak"].get("min", "N/A"),
            "GPU Temp Peak Max (°C)": metrics["gpu_temp_peak"].get("max", "N/A"),
        }
        rows.append(row)
    
    if rows:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"✓ CSV report saved to {csv_file}")
    
    return csv_file


def main():
    """Main function to process computational logs."""
    # Define paths
    script_dir = Path(__file__).parent.parent.parent
    logs_dir = script_dir / "outputs" / "computation_logs"
    output_dir = script_dir / "evaluation_reports" / "computational"
    
    print(f"Loading logs from: {logs_dir}")
    
    # Load and process logs
    logs = load_log_files(logs_dir)
    if not logs:
        print("No log files found!")
        return
    
    print(f"Found {len(logs)} log files")
    
    # Aggregate and generate summary
    model_stats = aggregate_computational_logs(logs)
    summary = generate_summary(model_stats)
    
    # Save results
    print(f"\nSaving reports to: {output_dir}")
    save_to_json(summary, str(output_dir))
    save_to_csv(summary, str(output_dir))
    
    # Print summary
    print("\n" + "="*60)
    print("COMPUTATIONAL PERFORMANCE SUMMARY")
    print("="*60)
    
    for model, metrics in sorted(summary.items()):
        print(f"\n📊 {model}")
        print(f"   Success Rate: {metrics['success_rate']}% ({metrics['successful_records']}/{metrics['total_records']})")
        print(f"   Execution Time: {metrics['execution_time']['mean']}s ± {metrics['execution_time'].get('stdev', 'N/A')}s")
        print(f"   CPU Usage: {metrics['cpu_usage']['mean']}% ± {metrics['cpu_usage'].get('stdev', 'N/A')}%")
        print(f"   RAM Peak: {metrics['ram_peak']['mean']}MB ± {metrics['ram_peak'].get('stdev', 'N/A')}MB")
        print(f"   GPU Utilization: {metrics['gpu_utilization']['mean']}% ± {metrics['gpu_utilization'].get('stdev', 'N/A')}%")
        print(f"   GPU Memory Used: {metrics['gpu_memory_used']['mean']}MB ± {metrics['gpu_memory_used'].get('stdev', 'N/A')}MB")
        print(f"   GPU Temp Peak: {metrics['gpu_temp_peak']['mean']}°C ± {metrics['gpu_temp_peak'].get('stdev', 'N/A')}°C")


if __name__ == "__main__":
    main()
