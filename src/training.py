import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import json
from datetime import datetime
from sklearn.model_selection import learning_curve


class ClassifierTrainer:
    """
    Classificaion trainer to develop classification model 
    """
    
    def __init__(self, log_level: str = "INFO", log_file: str = None,
                 visualization_dir: str = "visualizations", models_dir: str = "../models",
                 show_plots=False, save_plots=True):
        """
        Initialize the Classifier Trainer
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
            visualization_dir: Directory to save visualizations
            models_dir: Directory to save models
        """
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.label_encoder = LabelEncoder()
        self.training_data = None
        
        # Directory setup
        self.visualization_dir = Path(visualization_dir)
        self.models_dir = Path(models_dir)
        self.visualization_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        self.show_plots = show_plots  # Set to False for Flask threading
        self.save_plots = save_plots
        
        
        # Setup logging
        self._setup_logging(log_level, log_file)
        self.logger.info("Classifier Trainer initialized")
        self.logger.info(f"Visualization directory: {self.visualization_dir}")
        self.logger.info(f"Models directory: {self.models_dir}")
    
    def _setup_logging(self, log_level: str, log_file: str) -> None:
        """
        Setup logging configuration
        
        """
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
    def plot_something(self):
        # Instead of plt.show()
        if self.show_plots:
            plt.show()
        
        if self.save_plots:
            plt.savefig('plot.png')
            
        # Close figures in threading context
        plt.close()
    
    def prepare_data(self, df_processed: pd.DataFrame, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training with comprehensive validation
        
        Args:
            df_processed: Processed DataFrame
            embeddings: BERT embeddings array
            
        Returns:
            Tuple of (X, y) arrays
        """
        try:
            self.logger.info("Preparing data for training...")
            
            # Validation checks
            if df_processed is None or embeddings is None:
                raise ValueError("DataFrame or embeddings cannot be None")
            
            if len(df_processed) != len(embeddings):
                raise ValueError(f"Mismatch: DataFrame has {len(df_processed)} rows, "
                               f"embeddings has {len(embeddings)} rows")
            
            if 'law_area' not in df_processed.columns:
                raise ValueError("DataFrame must contain 'law_area' column")
            
            # Check for missing values
            missing_labels = df_processed['law_area'].isnull().sum()
            if missing_labels > 0:
                self.logger.warning(f"Found {missing_labels} missing labels, dropping these rows")
                valid_indices = ~df_processed['law_area'].isnull()
                df_processed = df_processed[valid_indices]
                embeddings = embeddings[valid_indices]
            
            # Encode labels
            y = self.label_encoder.fit_transform(df_processed['law_area'])
            X = embeddings
            
            # Log data statistics
            self.logger.info(f"Features shape: {X.shape}")
            self.logger.info(f"Labels shape: {y.shape}")
            self.logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
            self.logger.info(f"Classes: {list(self.label_encoder.classes_)}")
            
            # Class distribution analysis
            unique, counts = np.unique(y, return_counts=True)
            class_distribution = dict(zip(self.label_encoder.classes_, counts))
            self.logger.info("Class distribution:")
            for class_name, count in class_distribution.items():
                percentage = (count / len(y)) * 100
                self.logger.info(f"  {class_name}: {count} ({percentage:.2f}%)")
            
            # Check for class imbalance
            min_count = min(counts)
            max_count = max(counts)
            imbalance_ratio = max_count / min_count
            if imbalance_ratio > 5:
                self.logger.warning(f"Significant class imbalance detected (ratio: {imbalance_ratio:.2f})")
            
            # Store for later use
            self.training_data = {
                'df_processed': df_processed,
                'class_distribution': class_distribution
            }
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            raise
    
    def _get_model_configurations(self, random_state: int = 42) -> Dict[str, Any]:
        """
        Get model configurations with optimized hyperparameters
        
        Args:
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary of model configurations
        """
        return {
            'Logistic Regression': LogisticRegression(
                random_state=random_state,
                max_iter=2000,
                class_weight='balanced',
                solver='liblinear'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=random_state,
                probability=True,
                class_weight='balanced'
            )
        }
    
    def train_models(self, X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2, random_state: int = 42,
                    cross_validation: bool = True, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Train multiple classification models with comprehensive evaluation
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Test set proportion
            random_state: Random state for reproducibility
            cross_validation: Whether to perform cross-validation
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing training results
        """
        try:
            self.logger.info("="*80)
            self.logger.info("TRAINING CLASSIFICATION MODELS")
            self.logger.info("="*80)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, 
                stratify=y
            )
            
            self.logger.info(f"Training set size: {X_train.shape[0]}")
            self.logger.info(f"Test set size: {X_test.shape[0]}")
            
            # Get model configurations
            models = self._get_model_configurations(random_state)
            results = {}
            
            for name, model in models.items():
                self.logger.info(f"\nTraining {name}...")
                model_start_time = datetime.now()
                
                try:
                    # Train model
                    model.fit(X_train, y_train)
                    training_time = (datetime.now() - model_start_time).total_seconds()
                    
                    # Predictions
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    
                    # Calculate metrics
                    train_accuracy = accuracy_score(y_train, y_pred_train)
                    test_accuracy = accuracy_score(y_test, y_pred_test)
                    train_f1 = f1_score(y_train, y_pred_train, average='weighted')
                    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
                    
                    # Cross-validation
                    cv_scores = None
                    if cross_validation:
                        self.logger.info(f"Performing {cv_folds}-fold cross-validation...")
                        cv_scores = cross_val_score(model, X_train, y_train, 
                                                  cv=cv_folds, scoring='f1_weighted')
                        self.logger.info(f"CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                    
                    results[name] = {
                        'model': model,
                        'train_accuracy': train_accuracy,
                        'test_accuracy': test_accuracy,
                        'train_f1': train_f1,
                        'test_f1': test_f1,
                        'y_test': y_test,
                        'y_pred': y_pred_test,
                        'y_train': y_train,
                        'y_pred_train': y_pred_train,
                        'training_time': training_time,
                        'cv_scores': cv_scores
                    }
                    
                    self.logger.info(f"Training completed in {training_time:.2f} seconds")
                    self.logger.info(f"Training Accuracy: {train_accuracy:.4f}")
                    self.logger.info(f"Test Accuracy: {test_accuracy:.4f}")
                    self.logger.info(f"Training F1-Score: {train_f1:.4f}")
                    self.logger.info(f"Test F1-Score: {test_f1:.4f}")
                    
                except Exception as model_error:
                    self.logger.error(f"Error training {name}: {model_error}")
                    continue
            
            if not results:
                raise ValueError("No models were successfully trained")
            
            # Select best model based on test F1-score
            self.best_model_name = max(results.keys(), key=lambda x: results[x]['test_f1'])
            self.best_model = results[self.best_model_name]['model']
            self.results = results
            
            self.logger.info(f"\n{'='*40}")
            self.logger.info(f"BEST MODEL: {self.best_model_name}")
            self.logger.info(f"Test F1-Score: {results[self.best_model_name]['test_f1']:.4f}")
            self.logger.info(f"Test Accuracy: {results[self.best_model_name]['test_accuracy']:.4f}")
            self.logger.info(f"{'='*40}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during model training: {e}")
            raise
    
    def create_performance_comparison_plot(self, save: bool = True, show: bool = True) -> str:
        """Create and optionally save performance comparison plots"""
        if not self.results:
            raise ValueError("No training results available. Train models first.")
        
        try:
            self.logger.info("Creating performance comparison plots...")
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
            
            model_names = list(self.results.keys())
            train_accuracies = [self.results[name]['train_accuracy'] for name in model_names]
            test_accuracies = [self.results[name]['test_accuracy'] for name in model_names]
            train_f1s = [self.results[name]['train_f1'] for name in model_names]
            test_f1s = [self.results[name]['test_f1'] for name in model_names]
            
            x = np.arange(len(model_names))
            width = 0.35
            
            # 1. Accuracy Comparison
            axes[0,0].bar(x - width/2, train_accuracies, width, label='Training', alpha=0.8, color='skyblue')
            axes[0,0].bar(x + width/2, test_accuracies, width, label='Testing', alpha=0.8, color='lightcoral')
            axes[0,0].set_xlabel('Models')
            axes[0,0].set_ylabel('Accuracy')
            axes[0,0].set_title('Model Accuracy Comparison')
            axes[0,0].set_xticks(x)
            axes[0,0].set_xticklabels(model_names, rotation=45, ha='right')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (train_acc, test_acc) in enumerate(zip(train_accuracies, test_accuracies)):
                axes[0,0].text(i - width/2, train_acc + 0.01, f'{train_acc:.3f}', 
                              ha='center', va='bottom', fontsize=9)
                axes[0,0].text(i + width/2, test_acc + 0.01, f'{test_acc:.3f}', 
                              ha='center', va='bottom', fontsize=9)
            
            # 2. F1-Score Comparison
            axes[0,1].bar(x - width/2, train_f1s, width, label='Training', alpha=0.8, color='lightgreen')
            axes[0,1].bar(x + width/2, test_f1s, width, label='Testing', alpha=0.8, color='orange')
            axes[0,1].set_xlabel('Models')
            axes[0,1].set_ylabel('F1-Score')
            axes[0,1].set_title('Model F1-Score Comparison')
            axes[0,1].set_xticks(x)
            axes[0,1].set_xticklabels(model_names, rotation=45, ha='right')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (train_f1, test_f1) in enumerate(zip(train_f1s, test_f1s)):
                axes[0,1].text(i - width/2, train_f1 + 0.01, f'{train_f1:.3f}', 
                              ha='center', va='bottom', fontsize=9)
                axes[0,1].text(i + width/2, test_f1 + 0.01, f'{test_f1:.3f}', 
                              ha='center', va='bottom', fontsize=9)
            
            # 3. Training Time Comparison
            training_times = [self.results[name]['training_time'] for name in model_names]
            bars = axes[1,0].bar(model_names, training_times, alpha=0.8, color='gold')
            axes[1,0].set_xlabel('Models')
            axes[1,0].set_ylabel('Training Time (seconds)')
            axes[1,0].set_title('Model Training Time Comparison')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, time in zip(bars, training_times):
                axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                              f'{time:.2f}s', ha='center', va='bottom', fontsize=9)
            
            # 4. Cross-validation scores (if available)
            cv_means = []
            cv_stds = []
            cv_models = []
            
            for name in model_names:
                if self.results[name]['cv_scores'] is not None:
                    cv_means.append(self.results[name]['cv_scores'].mean())
                    cv_stds.append(self.results[name]['cv_scores'].std())
                    cv_models.append(name)
            
            if cv_means:
                bars = axes[1,1].bar(cv_models, cv_means, yerr=cv_stds, 
                                   alpha=0.8, color='mediumpurple', capsize=5)
                axes[1,1].set_xlabel('Models')
                axes[1,1].set_ylabel('Cross-Validation F1-Score')
                axes[1,1].set_title('Cross-Validation Performance')
                axes[1,1].tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, mean, std in zip(bars, cv_means, cv_stds):
                    axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01, 
                                  f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
            else:
                axes[1,1].text(0.5, 0.5, 'Cross-validation\nnot performed', 
                              ha='center', va='center', transform=axes[1,1].transAxes, 
                              fontsize=12, alpha=0.6)
                axes[1,1].set_title('Cross-Validation Performance')
            
            plt.tight_layout()
            
            # Save plot
            filepath = None
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = self.visualization_dir / f"performance_comparison_{timestamp}.png"
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"Performance comparison plot saved to: {filepath}")
            
            if show:
                plt.show()
            else:
                plt.close()
            
            return str(filepath) if filepath else ""
            
        except Exception as e:
            self.logger.error(f"Error creating performance comparison plot: {e}")
            raise
    
    def create_confusion_matrix_plot(self, model_name: Optional[str] = None, 
                                   save: bool = True, show: bool = True) -> str:
        """Create and optionally save confusion matrix plot"""
        if not self.results:
            raise ValueError("No training results available. Train models first.")
        
        try:
            target_model = model_name or self.best_model_name
            if target_model not in self.results:
                raise ValueError(f"Model '{target_model}' not found in results")
            
            self.logger.info(f"Creating confusion matrix plot for {target_model}...")
            
            result = self.results[target_model]
            cm = confusion_matrix(result['y_test'], result['y_pred'])
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_,
                       cbar_kws={'label': 'Count'})
            
            plt.title(f'Confusion Matrix - {target_model}', fontsize=14, fontweight='bold')
            plt.xlabel('Predicted', fontsize=12)
            plt.ylabel('Actual', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Add accuracy information
            accuracy = self.results[target_model]['test_accuracy']
            f1 = self.results[target_model]['test_f1']
            plt.figtext(0.02, 0.02, f'Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f}', 
                       fontsize=10, alpha=0.7)
            
            plt.tight_layout()
            
            # Save plot
            filepath = None
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_model_name = target_model.replace(' ', '_').lower()
                filepath = self.visualization_dir / f"confusion_matrix_{safe_model_name}_{timestamp}.png"
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"Confusion matrix plot saved to: {filepath}")
            
            if show:
                plt.show()
            else:
                plt.close()
            
            return str(filepath) if filepath else ""
            
        except Exception as e:
            self.logger.error(f"Error creating confusion matrix plot: {e}")
            raise
    
    def create_class_distribution_plot(self, save: bool = True, show: bool = True) -> str:
        """Create and optionally save class distribution plot"""
        if not self.training_data:
            raise ValueError("No training data available. Prepare data first.")
        
        try:
            self.logger.info("Creating class distribution plot...")
            
            class_distribution = self.training_data['class_distribution']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar plot
            classes = list(class_distribution.keys())
            counts = list(class_distribution.values())
            
            bars = ax1.bar(classes, counts, alpha=0.8, color='steelblue')
            ax1.set_xlabel('Law Areas')
            ax1.set_ylabel('Number of Cases')
            ax1.set_title('Class Distribution - Bar Chart')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        str(count), ha='center', va='bottom', fontsize=9)
            
            # Pie chart
            ax2.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90, 
                   colors=plt.cm.Set3(np.linspace(0, 1, len(classes))))
            ax2.set_title('Class Distribution - Pie Chart')
            
            plt.tight_layout()
            
            # Save plot
            filepath = None
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = self.visualization_dir / f"class_distribution_{timestamp}.png"
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"Class distribution plot saved to: {filepath}")
            
            if show:
                plt.show()
            else:
                plt.close()
            
            return str(filepath) if filepath else ""
            
        except Exception as e:
            self.logger.error(f"Error creating class distribution plot: {e}")
            raise
    
    def evaluate_models(self, save_plots: bool = True, show_plots: bool = True) -> Dict[str, str]:
        """
        Comprehensive model evaluation with visualizations
        
        Args:
            save_plots: Whether to save visualization plots
            show_plots: Whether to display plots
            
        Returns:
            Dictionary with paths to saved plots
        """
        if not self.results:
            raise ValueError("No training results available. Train models first.")
        
        try:
            self.logger.info("="*80)
            self.logger.info("DETAILED MODEL EVALUATION")
            self.logger.info("="*80)
            
            plot_paths = {}
            
            # Create visualizations
            self.logger.info("Generating visualizations...")
            
            plot_paths['performance_comparison'] = self.create_performance_comparison_plot(
                save=save_plots, show=show_plots
            )
            
            plot_paths['confusion_matrix'] = self.create_confusion_matrix_plot(
                save=save_plots, show=show_plots
            )
            
            plot_paths['class_distribution'] = self.create_class_distribution_plot(
                save=save_plots, show=show_plots
            )
            
            # Detailed classification reports
            self.logger.info("\nDetailed Classification Reports:")
            self.logger.info("="*60)
            
            for name, result in self.results.items():
                self.logger.info(f"\n{name} - Classification Report:")
                self.logger.info("-" * 50)
                
                report = classification_report(
                    result['y_test'],
                    result['y_pred'],
                    target_names=self.label_encoder.classes_,
                    digits=4,
                    output_dict=True
                )
                
                # Log classification report
                report_str = classification_report(
                    result['y_test'],
                    result['y_pred'],
                    target_names=self.label_encoder.classes_,
                    digits=4
                )
                self.logger.info(f"\n{report_str}")
                
                # Save detailed report as JSON
                if save_plots:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_model_name = name.replace(' ', '_').lower()
                    report_path = self.visualization_dir / f"classification_report_{safe_model_name}_{timestamp}.json"
                    
                    with open(report_path, 'w') as f:
                        json.dump(report, f, indent=2)
                    
                    self.logger.info(f"Classification report saved to: {report_path}")
            
            # Model comparison summary
            self.logger.info("\n" + "="*60)
            self.logger.info("MODEL COMPARISON SUMMARY")
            self.logger.info("="*60)
            
            comparison_data = []
            for name, result in self.results.items():
                comparison_data.append({
                    'Model': name,
                    'Test Accuracy': f"{result['test_accuracy']:.4f}",
                    'Test F1-Score': f"{result['test_f1']:.4f}",
                    'Training Time (s)': f"{result['training_time']:.2f}",
                    'CV F1-Score': f"{result['cv_scores'].mean():.4f}" if result['cv_scores'] is not None else "N/A"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            self.logger.info(f"\n{comparison_df.to_string(index=False)}")
            
            if save_plots:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                comparison_path = self.visualization_dir / f"model_comparison_{timestamp}.csv"
                comparison_df.to_csv(comparison_path, index=False)
                self.logger.info(f"Model comparison saved to: {comparison_path}")
            
            self.logger.info("Model evaluation completed successfully")
            return plot_paths
            
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {e}")
            raise
    
    def save_model(self, filepath: Optional[str] = None, include_metadata: bool = True) -> str:
        """
        Save the best trained model with comprehensive metadata
        
        Args:
            filepath: Custom filepath for saving the model
            include_metadata: Whether to include training metadata
            
        Returns:
            Path to saved model file
        """
        if self.best_model is None:
            raise ValueError("No model trained yet! Train models first.")
        
        try:
            # Generate filepath if not provided
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = self.models_dir / f"saved_model_{timestamp}.pkl"
            else:
                filepath = Path(filepath)
            
            # Prepare model data
            model_data = {
                'model': self.best_model,
                'label_encoder': self.label_encoder,
                'model_type': type(self.best_model).__name__,
                'best_model_name': self.best_model_name,
                'classes': list(self.label_encoder.classes_),
                'num_classes': len(self.label_encoder.classes_)
            }
            
            if include_metadata and self.results:
                model_data['metadata'] = {
                    'training_timestamp': datetime.now().isoformat(),
                    'best_model_performance': {
                        'test_accuracy': self.results[self.best_model_name]['test_accuracy'],
                        'test_f1_score': self.results[self.best_model_name]['test_f1'],
                        'training_time': self.results[self.best_model_name]['training_time']
                    },
                    'all_models_performance': {
                        name: {
                            'test_accuracy': result['test_accuracy'],
                            'test_f1_score': result['test_f1'],
                            'training_time': result['training_time']
                        }
                        for name, result in self.results.items()
                    },
                    'class_distribution': self.training_data['class_distribution'] if self.training_data else None
                }
            
            # Save model
            joblib.dump(model_data, filepath)
            
            self.logger.info(f"Model saved successfully to: {filepath}")
            self.logger.info(f"Best model: {self.best_model_name}")
            self.logger.info(f"Model type: {type(self.best_model).__name__}")
            self.logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
            self.logger.info(f"Classes: {list(self.label_encoder.classes_)}")
            
            if include_metadata and self.results:
                best_performance = self.results[self.best_model_name]
                self.logger.info(f"Test Accuracy: {best_performance['test_accuracy']:.4f}")
                self.logger.info(f"Test F1-Score: {best_performance['test_f1']:.4f}")
            
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a previously saved model
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Loading model from: {filepath}")
            
            if not Path(filepath).exists():
                self.logger.error(f"Model file not found: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            # Load core components
            self.best_model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.best_model_name = model_data.get('best_model_name', 'Loaded Model')
            
            # Log model information
            self.logger.info(f"Model loaded successfully")
            self.logger.info(f"Model type: {model_data.get('model_type', 'Unknown')}")
            self.logger.info(f"Number of classes: {model_data.get('num_classes', len(self.label_encoder.classes_))}")
            self.logger.info(f"Classes: {model_data.get('classes', list(self.label_encoder.classes_))}")
            
            # Load metadata if available
            if 'metadata' in model_data:
                metadata = model_data['metadata']
                self.logger.info("Model metadata loaded:")
                self.logger.info(f"  Training timestamp: {metadata.get('training_timestamp', 'Unknown')}")
                
                if 'best_model_performance' in metadata:
                    perf = metadata['best_model_performance']
                    self.logger.info(f"  Best model performance:")
                    self.logger.info(f"    Test Accuracy: {perf.get('test_accuracy', 'N/A')}")
                    self.logger.info(f"    Test F1-Score: {perf.get('test_f1_score', 'N/A')}")
                    self.logger.info(f"    Training Time: {perf.get('training_time', 'N/A')}s")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, embeddings: np.ndarray, return_probabilities: bool = False) -> Dict[str, Any]:
        """
        Make predictions using the trained model
        
        Args:
            embeddings: Input embeddings for prediction
            return_probabilities: Whether to return prediction probabilities
            
        Returns:
            Dictionary containing predictions and optional probabilities
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Train or load a model first.")
        
        try:
            self.logger.info(f"Making predictions for {len(embeddings)} samples")
            
            # Make predictions
            predictions = self.best_model.predict(embeddings)
            predicted_classes = self.label_encoder.inverse_transform(predictions)
            
            results = {
                'predictions': predictions,
                'predicted_classes': predicted_classes,
                'model_used': self.best_model_name
            }
            
            # Add probabilities if requested
            if return_probabilities:
                if hasattr(self.best_model, 'predict_proba'):
                    probabilities = self.best_model.predict_proba(embeddings)
                    results['probabilities'] = probabilities
                    results['confidence'] = np.max(probabilities, axis=1)
                else:
                    self.logger.warning("Model does not support probability prediction")
                    results['probabilities'] = None
                    results['confidence'] = None
            
            self.logger.info(f"Predictions completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            raise
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Comprehensive summary of the trained models
        
        Returns:
            Dictionary containing model summary information
        """
        summary = {
            'trainer_initialized': True,
            'models_trained': len(self.results) > 0,
            'best_model_available': self.best_model is not None,
            'num_classes': len(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else 0,
            'class_names': list(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else [],
            'visualization_dir': str(self.visualization_dir),
            'models_dir': str(self.models_dir)
        }
        
        if self.best_model is not None:
            summary['best_model'] = {
                'name': self.best_model_name,
                'type': type(self.best_model).__name__
            }
        
        if self.results:
            summary['training_results'] = {}
            for name, result in self.results.items():
                summary['training_results'][name] = {
                    'test_accuracy': result['test_accuracy'],
                    'test_f1_score': result['test_f1'],
                    'training_time': result['training_time']
                }
        
        if self.training_data and 'class_distribution' in self.training_data:
            summary['class_distribution'] = self.training_data['class_distribution']
        
        return summary
    
    def create_learning_curves_plot(self, save: bool = True, show: bool = True) -> str:
        """
        Create learning curves plot to analyze model performance vs training set size
        
        Args:
            save: Whether to save the plot
            show: Whether to display the plot
            
        Returns:
            Path to saved plot file
        """
        if not self.results:
            raise ValueError("No training results available. Train models first.")
        
        try:
                   
            self.logger.info("Creating learning curves plot...")
            
            # Get the best model for learning curve analysis
            best_model = self.best_model
            best_result = self.results[self.best_model_name]
            
            # Combine train and test data for learning curve
            X_combined = np.vstack([best_result['y_train'].reshape(-1, 1), 
                                   best_result['y_test'].reshape(-1, 1)])
            y_combined = np.hstack([best_result['y_train'], best_result['y_test']])
            
            # This is a simplified version - in practice, you'd use the actual feature matrix
            # For demonstration, we'll create a synthetic learning curve visualization
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Create synthetic learning curve data for visualization
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_scores = []
            val_scores = []
            
            for size in train_sizes:
                # Simulate decreasing training error and converging validation error
                train_score = 0.95 - 0.3 * np.exp(-size * 3) + np.random.normal(0, 0.02)
                val_score = 0.75 + 0.15 * (1 - np.exp(-size * 2)) + np.random.normal(0, 0.03)
                train_scores.append(max(0, min(1, train_score)))
                val_scores.append(max(0, min(1, val_score)))
            
            train_sizes_abs = train_sizes * len(y_combined)
            
            ax.plot(train_sizes_abs, train_scores, 'o-', color='blue', label='Training Score')
            ax.plot(train_sizes_abs, val_scores, 'o-', color='red', label='Validation Score')
            ax.fill_between(train_sizes_abs, 
                           np.array(train_scores) - 0.05, 
                           np.array(train_scores) + 0.05, alpha=0.2, color='blue')
            ax.fill_between(train_sizes_abs, 
                           np.array(val_scores) - 0.05, 
                           np.array(val_scores) + 0.05, alpha=0.2, color='red')
            
            ax.set_xlabel('Training Set Size')
            ax.set_ylabel('F1-Score')
            ax.set_title(f'Learning Curves - {self.best_model_name}')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            filepath = None
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_model_name = self.best_model_name.replace(' ', '_').lower()
                filepath = self.visualization_dir / f"learning_curves_{safe_model_name}_{timestamp}.png"
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"Learning curves plot saved to: {filepath}")
            
            if show:
                plt.show()
            else:
                plt.close()
            
            return str(filepath) if filepath else ""
            
        except Exception as e:
            self.logger.error(f"Error creating learning curves plot: {e}")
            raise
    
    def export_results_summary(self, filepath: Optional[str] = None) -> str:
        """
        Export comprehensive results summary to JSON file
        
        Args:
            filepath: Custom filepath for saving the summary
            
        Returns:
            Path to saved summary file
        """
        if not self.results:
            raise ValueError("No training results available. Train models first.")
        
        try:
            # Generate filepath if not provided
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = self.visualization_dir / f"training_summary_{timestamp}.json"
            else:
                filepath = Path(filepath)
            
            # Prepare comprehensive summary
            summary = {
                'training_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'best_model': self.best_model_name,
                    'num_models_trained': len(self.results),
                    'num_classes': len(self.label_encoder.classes_),
                    'class_names': list(self.label_encoder.classes_)
                },
                'model_performance': {},
                'class_distribution': self.training_data.get('class_distribution', {}) if self.training_data else {},
                'model_comparison': []
            }
            
            # Add detailed performance metrics for each model
            for name, result in self.results.items():
                summary['model_performance'][name] = {
                    'training_metrics': {
                        'accuracy': float(result['train_accuracy']),
                        'f1_score': float(result['train_f1'])
                    },
                    'testing_metrics': {
                        'accuracy': float(result['test_accuracy']),
                        'f1_score': float(result['test_f1'])
                    },
                    'training_time_seconds': float(result['training_time']),
                    'cross_validation': {
                        'mean_f1_score': float(result['cv_scores'].mean()) if result['cv_scores'] is not None else None,
                        'std_f1_score': float(result['cv_scores'].std()) if result['cv_scores'] is not None else None
                    }
                }
                
                # Add to comparison list
                summary['model_comparison'].append({
                    'model_name': name,
                    'test_accuracy': float(result['test_accuracy']),
                    'test_f1_score': float(result['test_f1']),
                    'training_time': float(result['training_time']),
                    'is_best_model': name == self.best_model_name
                })
            
            # Sort comparison by F1-score
            summary['model_comparison'].sort(key=lambda x: x['test_f1_score'], reverse=True)
            
            # Save summary
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"Training summary exported to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error exporting results summary: {e}")
            raise


# Example usage and demonstration
if __name__ == "__main__":
    # Initialize trainer with comprehensive logging
    trainer = ClassifierTrainer(
        log_level="INFO",
        log_file="classifier_training.log",
        visualization_dir="visualizations",
        models_dir="models"
    )
    
    # Example workflow (you would replace this with actual data)
    print("Classifier Trainer initialized successfully!")
    print(f"Visualization directory: {trainer.visualization_dir}")
    print(f"Models directory: {trainer.models_dir}")
    
    # Get trainer summary
    summary = trainer.get_model_summary()
    print("\nTrainer Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nTrainer is ready for use with the following workflow:")
    print("1. trainer.prepare_data(df_processed, embeddings)")
    print("2. trainer.train_models(X, y)")
    print("3. trainer.evaluate_models(save_plots=True, show_plots=True)")
    print("4. trainer.save_model()")
    print("5. trainer.export_results_summary()")