from config import ERRORS, LEFT, RIGHT, SCORING, OUTPUT

class StatsManager:
    def __init__(self):
        """Initialize the statistics manager with empty counters and results."""
        self.stats = {
            ERRORS['invalid_ellipse']: {"count": 0, "point": SCORING['invalid_ellipse']},
            ERRORS['invalid_sign']: {"count": 0, "point": SCORING['invalid_sign']},
            ERRORS['no_sign_detected']: {"count": 0, "point": SCORING['no_sign_detected']},
            LEFT: {"count": 0, "point": SCORING['left']},
            RIGHT: {"count": 0, "point": SCORING['right']},
            ERRORS['image_read_error']: {"count": 0, "point": SCORING['image_read_error']},
            "Total Processed": 0,
            "Total Score": 0
        }
        self.results = {}

    def increment_total_processed(self):
        self.stats["Total Processed"] += 1

    def add_result(self, file_name, result):
        self.results[file_name] = result
        if result in self.stats:
            self.stats[result]["count"] += 1

    def calculate_total_score(self):
        self.stats["Total Score"] = sum(
            value["count"] * value["point"] 
            for key, value in self.stats.items() 
            if key not in ["Total Processed", "Total Score"]
        )

    def write_to_file(self, output_file_path):
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(OUTPUT['stats']['title'] + "\n")
                f.write(OUTPUT['stats']['total_processed'].format(self.stats['Total Processed']) + "\n")
                
                for key, value in self.stats.items():
                    if key not in ["Total Processed", "Total Score"]:
                        f.write(OUTPUT['stats']['score_format'].format(
                            key, value['count'], value['count'] * value['point']) + "\n")
                
                f.write(OUTPUT['stats']['total_score'].format(self.stats['Total Score']) + "\n")
                f.write("\n" + OUTPUT['stats']['results_title'] + "\n")

                for file_name, result in self.results.items():
                    f.write(f"{file_name}: {result}\n")

            print(OUTPUT['processing_completed'].format(output_file_path))
        except IOError as e:
            print(f"Error writing results to file: {str(e)}")
            raise

    def get_stats(self):
        return self.stats

    def get_results(self):
        return self.results 