import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta

from layer_mvp_0022 import (
    WikipediaVaccineSearchAPI,
    ClinicalTrialsAPI,
    GrangerCausalityAnalyzer,
    MarketOpportunityReportGenerator,
    VaccineResearchTracker
)


class TestWikipediaVaccineSearchAPI:
    """Unit tests for Wikipedia vaccine search volume data API endpoint."""
    
    def test_get_vaccine_search_volume_data_returns_valid_response(self):
        """Test that Wikipedia API returns properly formatted search volume data."""
        api = WikipediaVaccineSearchAPI()
        with patch('layer_mvp_0022.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                'items': [
                    {'timestamp': '2023010100', 'views': 1500},
                    {'timestamp': '2023010200', 'views': 1650}
                ]
            }
            mock_get.return_value = mock_response
            
            result = api.get_vaccine_search_volume_data('COVID-19_vaccine', '2023-01-01', '2023-01-02')
            
            assert isinstance(result, dict)
            assert 'search_volumes' in result
            assert len(result['search_volumes']) == 2
            assert result['search_volumes'][0]['views'] == 1500

    def test_get_vaccine_search_volume_data_handles_date_range(self):
        """Test that Wikipedia API correctly processes date range parameters."""
        api = WikipediaVaccineSearchAPI()
        start_date = '2023-01-01'
        end_date = '2023-01-31'
        
        with patch('layer_mvp_0022.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {'items': []}
            mock_get.return_value = mock_response
            
            api.get_vaccine_search_volume_data('COVID-19_vaccine', start_date, end_date)
            
            call_args = mock_get.call_args
            assert start_date.replace('-', '') in call_args[0][0]
            assert end_date.replace('-', '') in call_args[0][0]


class TestClinicalTrialsAPI:
    """Unit tests for COVID-19 clinical trials count data API endpoint."""
    
    def test_get_covid_clinical_trials_count_returns_valid_data(self):
        """Test that clinical trials API returns properly formatted count data."""
        api = ClinicalTrialsAPI()
        with patch('layer_mvp_0022.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                'studies': [
                    {'nct_id': 'NCT123', 'start_date': '2023-01-01', 'status': 'Active'},
                    {'nct_id': 'NCT456', 'start_date': '2023-01-02', 'status': 'Recruiting'}
                ]
            }
            mock_get.return_value = mock_response
            
            result = api.get_covid_clinical_trials_count('2023-01-01', '2023-01-31')
            
            assert isinstance(result, dict)
            assert 'trials_count' in result
            assert 'daily_counts' in result
            assert result['trials_count'] == 2

    def test_get_covid_clinical_trials_count_filters_by_date(self):
        """Test that clinical trials API correctly filters trials by date range."""
        api = ClinicalTrialsAPI()
        with patch('layer_mvp_0022.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                'studies': [
                    {'nct_id': 'NCT123', 'start_date': '2023-01-15', 'status': 'Active'}
                ]
            }
            mock_get.return_value = mock_response
            
            result = api.get_covid_clinical_trials_count('2023-01-01', '2023-01-31')
            
            assert len(result['daily_counts']) > 0
            assert '2023-01-15' in str(result['daily_counts'])


class TestGrangerCausalityAnalyzer:
    """Unit tests for Granger causality correlation calculation between datasets."""
    
    def test_calculate_granger_causality_returns_correlation_metrics(self):
        """Test that Granger causality analyzer returns proper correlation metrics."""
        analyzer = GrangerCausalityAnalyzer()
        
        search_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=30),
            'search_volume': range(100, 130)
        })
        
        trials_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=30),
            'trial_count': range(10, 40)
        })
        
        result = analyzer.calculate_granger_causality(search_data, trials_data)
        
        assert isinstance(result, dict)
        assert 'p_value' in result
        assert 'f_statistic' in result
        assert 'causality_direction' in result
        assert 0 <= result['p_value'] <= 1

    def test_calculate_granger_causality_handles_lag_parameter(self):
        """Test that Granger causality analyzer properly uses lag parameter."""
        analyzer = GrangerCausalityAnalyzer()
        
        search_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50),
            'search_volume': range(100, 150)
        })
        
        trials_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50),
            'trial_count': range(20, 70)
        })
        
        result_lag_5 = analyzer.calculate_granger_causality(search_data, trials_data, max_lag=5)
        result_lag_10 = analyzer.calculate_granger_causality(search_data, trials_data, max_lag=10)
        
        assert result_lag_5['max_lag_used'] == 5
        assert result_lag_10['max_lag_used'] == 10


class TestMarketOpportunityReportGenerator:
    """Unit tests for market opportunity assessment report generation."""
    
    def test_generate_market_opportunity_report_creates_comprehensive_analysis(self):
        """Test that report generator creates comprehensive market opportunity analysis."""
        generator = MarketOpportunityReportGenerator()
        
        correlation_data = {
            'p_value': 0.05,
            'f_statistic': 4.2,
            'causality_direction': 'search_to_trials',
            'correlation_coefficient': 0.75
        }
        
        search_trends = {
            'total_searches': 50000,
            'trend_direction': 'increasing',
            'peak_dates': ['2023-01-15', '2023-01-25']
        }
        
        trials_summary = {
            'total_trials': 120,
            'active_trials': 80,
            'completion_rate': 0.85
        }
        
        report = generator.generate_market_opportunity_report(
            correlation_data, search_trends, trials_summary
        )
        
        assert isinstance(report, dict)
        assert 'market_opportunity_score' in report
        assert 'investment_recommendation' in report
        assert 'risk_assessment' in report
        assert 'key_insights' in report
        assert 0 <= report['market_opportunity_score'] <= 100

    def test_generate_market_opportunity_report_includes_statistical_significance(self):
        """Test that report includes statistical significance assessment."""
        generator = MarketOpportunityReportGenerator()
        
        significant_correlation = {
            'p_value': 0.01,
            'f_statistic': 8.5,
            'causality_direction': 'bidirectional',
            'correlation_coefficient': 0.82
        }
        
        insignificant_correlation = {
            'p_value': 0.15,
            'f_statistic': 2.1,
            'causality_direction': 'none',
            'correlation_coefficient': 0.25
        }
        
        search_trends = {'total_searches': 10000, 'trend_direction': 'stable'}
        trials_summary = {'total_trials': 50, 'active_trials': 30}
        
        significant_report = generator.generate_market_opportunity_report(
            significant_correlation, search_trends, trials_summary
        )
        
        insignificant_report = generator.generate_market_opportunity_report(
            insignificant_correlation, search_trends, trials_summary
        )
        
        assert significant_report['statistical_significance'] == 'significant'
        assert insignificant_report['statistical_significance'] == 'not_significant'
        assert significant_report['market_opportunity_score'] > insignificant_report['market_opportunity_score']


class TestIntegrationDataPipeline:
    """Integration tests for Wikipedia API and clinical trials data source integration with correlation analysis."""
    
    def test_wikipedia_data_fetch_and_processing(self):
        """Test Wikipedia data fetching and processing integration with correlation analysis."""
        wikipedia_api = WikipediaVaccineSearchAPI()
        analyzer = GrangerCausalityAnalyzer()
        
        with patch('layer_mvp_0022.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                'items': [
                    {'timestamp': '2023010100', 'views': 1000},
                    {'timestamp': '2023010200', 'views': 1200},
                    {'timestamp': '2023010300', 'views': 1100}
                ]
            }
            mock_get.return_value = mock_response
            
            search_data = wikipedia_api.get_vaccine_search_volume_data(
                'COVID-19_vaccine', '2023-01-01', '2023-01-03'
            )
            
            processed_data = analyzer.preprocess_search_data(search_data)
            
            assert isinstance(processed_data, pd.DataFrame)
            assert 'date' in processed_data.columns
            assert 'search_volume' in processed_data.columns
            assert len(processed_data) == 3

    def test_clinical_trials_data_correlation(self):
        """Test clinical trials data integration with correlation analysis workflow."""
        trials_api = ClinicalTrialsAPI()
        analyzer = GrangerCausalityAnalyzer()
        
        with patch('layer_mvp_0022.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                'studies': [
                    {'nct_id': 'NCT001', 'start_date': '2023-01-01', 'status': 'Active'},
                    {'nct_id': 'NCT002', 'start_date': '2023-01-02', 'status': 'Recruiting'},
                    {'nct_id': 'NCT003', 'start_date': '2023-01-03', 'status': 'Active'}
                ]
            }
            mock_get.return_value = mock_response
            
            trials_data = trials_api.get_covid_clinical_trials_count('2023-01-01', '2023-01-31')
            processed_trials = analyzer.preprocess_trials_data(trials_data)
            
            mock_search_data = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=10),
                'search_volume': range(1000, 1010)
            })
            
            correlation_result = analyzer.calculate_granger_causality(
                mock_search_data, processed_trials
            )
            
            assert isinstance(correlation_result, dict)
            assert 'p_value' in correlation_result
            assert processed_trials is not None


class TestE2EMarketAnalysis:
    """End-to-end tests for complete market analysis workflow from data collection to report generation."""
    
    def test_complete_vaccine_research_analysis_pipeline(self):
        """Test end-to-end workflow from data collection to market opportunity report generation."""
        tracker = VaccineResearchTracker()
        
        with patch('layer_mvp_0022.WikipediaVaccineSearchAPI.get_vaccine_search_volume_data') as mock_search, \
             patch('layer_mvp_0022.ClinicalTrialsAPI.get_covid_clinical_trials_count') as mock_trials:
            
            mock_search.return_value = {
                'search_volumes': [
                    {'timestamp': '2023010100', 'views': 1500},
                    {'timestamp': '2023010200', 'views': 1650},
                    {'timestamp': '2023010300', 'views': 1400}
                ]
            }
            
            mock_trials.return_value = {
                'trials_count': 25,
                'daily_counts': [
                    {'date': '2023-01-01', 'count': 8},
                    {'date': '2023-01-02', 'count': 10},
                    {'date': '2023-01-03', 'count': 7}
                ]
            }
            
            analysis_result = tracker.run_complete_analysis(
                vaccine_term='COVID-19_vaccine',
                start_date='2023-01-01',
                end_date='2023-01-31'
            )
            
            assert isinstance(analysis_result, dict)
            assert 'search_data_summary' in analysis_result
            assert 'trials_data_summary' in analysis_result
            assert 'granger_causality_results' in analysis_result
            assert 'market_opportunity_report' in analysis_result
            
            market_report = analysis_result['market_opportunity_report']
            assert 'market_opportunity_score' in market_report
            assert 'investment_recommendation' in market_report
            assert 'statistical_significance' in market_report
            
            granger_results = analysis_result['granger_causality_results']
            assert 'p_value' in granger_results
            assert 'causality_direction' in granger_results
            
            assert analysis_result['search_data_summary']['total_data_points'] == 3
            assert analysis_result['trials_data_summary']['total_trials'] == 25
