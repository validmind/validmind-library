% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/platform.R
\name{display_report}
\alias{display_report}
\title{Produce RMarkdown-compatible output of all results}
\usage{
display_report(processed_results)
}
\arguments{
\item{processed_results}{A list of processed result objects}
}
\value{
A formatted list of RMarkdown widgets
}
\description{
Produce RMarkdown-compatible output of all results
}
\examples{
\dontrun{
vm_dataset = vm_r$init_dataset(
  dataset=data,
  target_column="Exited",
  class_labels=list("0" = "Did not exit", "1" = "Exited")
)

tabular_suite_results <- vm_r$run_test_suite("tabular_dataset", dataset=vm_dataset)

processed_results <- process_result(tabular_suite_results)
all_widgets <- display_report(processed_results)
for (widget in all_widgets) {
  print(widget)
}
}

}
