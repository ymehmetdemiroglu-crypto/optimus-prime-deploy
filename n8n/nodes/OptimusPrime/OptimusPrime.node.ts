import {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
	NodeOperationError,
} from 'n8n-workflow';

export class OptimusPrime implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Optimus Prime',
		name: 'optimusPrime',
		icon: 'file:optimusPrime.svg',
		group: ['transform'],
		version: 1,
		subtitle: '={{$parameter["resource"] + ": " + $parameter["operation"]}}',
		description: 'Interact with the Optimus Prime AI-powered Amazon PPC War Room',
		defaults: { name: 'Optimus Prime' },
		inputs: ['main'],
		outputs: ['main'],
		credentials: [
			{
				name: 'optimusPrimeApi',
				required: true,
			},
		],
		properties: [
			// ──────────────────────────────────────────────
			//  Resource selector
			// ──────────────────────────────────────────────
			{
				displayName: 'Resource',
				name: 'resource',
				type: 'options',
				noDataExpression: true,
				options: [
					{ name: 'Dashboard', value: 'dashboard' },
					{ name: 'Campaign', value: 'campaign' },
					{ name: 'Anomaly', value: 'anomaly' },
					{ name: 'Optimization', value: 'optimization' },
					{ name: 'Contextual Bidding', value: 'contextualBidding' },
					{ name: 'RL Budget', value: 'rlBudget' },
					{ name: 'Semantic Intelligence', value: 'semantic' },
					{ name: 'Competitive Intel', value: 'competitive' },
					{ name: 'Ingestion', value: 'ingestion' },
					{ name: 'Health', value: 'health' },
				],
				default: 'dashboard',
			},

			// ──────────────────────────────────────────────
			//  Dashboard operations
			// ──────────────────────────────────────────────
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				displayOptions: { show: { resource: ['dashboard'] } },
				options: [
					{ name: 'Get Summary', value: 'getSummary', description: 'Fetch War Room KPIs' },
					{ name: 'Get Chart Data', value: 'getChartData', description: 'Fetch time-series chart data' },
					{ name: 'Get AI Actions', value: 'getAiActions', description: 'Fetch AI optimization actions' },
					{ name: 'Get Client Matrix', value: 'getClientMatrix', description: 'Fetch global client matrix' },
					{ name: 'Get Client Dashboard', value: 'getClientDashboard', description: 'Fetch client-specific dashboard' },
				],
				default: 'getSummary',
			},
			{
				displayName: 'Time Range',
				name: 'timeRange',
				type: 'options',
				displayOptions: { show: { resource: ['dashboard'], operation: ['getChartData'] } },
				options: [
					{ name: '7 Days', value: '7d' },
					{ name: '30 Days', value: '30d' },
					{ name: 'Year to Date', value: 'ytd' },
				],
				default: '7d',
			},
			{
				displayName: 'Client ID',
				name: 'clientId',
				type: 'string',
				displayOptions: { show: { resource: ['dashboard'], operation: ['getClientDashboard'] } },
				default: '',
				required: true,
				description: 'The client identifier (e.g. NX-4029)',
			},

			// ──────────────────────────────────────────────
			//  Campaign operations
			// ──────────────────────────────────────────────
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				displayOptions: { show: { resource: ['campaign'] } },
				options: [
					{ name: 'List All', value: 'list', description: 'List all campaigns' },
					{ name: 'Update Strategy', value: 'updateStrategy', description: 'Change a campaign AI strategy' },
				],
				default: 'list',
			},
			{
				displayName: 'Campaign ID',
				name: 'campaignId',
				type: 'string',
				displayOptions: { show: { resource: ['campaign'], operation: ['updateStrategy'] } },
				default: '',
				required: true,
			},
			{
				displayName: 'AI Mode',
				name: 'aiMode',
				type: 'options',
				displayOptions: { show: { resource: ['campaign'], operation: ['updateStrategy'] } },
				options: [
					{ name: 'Manual', value: 'manual' },
					{ name: 'Auto Pilot', value: 'auto_pilot' },
					{ name: 'Aggressive Growth', value: 'aggressive_growth' },
					{ name: 'Profit Guard', value: 'profit_guard' },
				],
				default: 'auto_pilot',
			},

			// ──────────────────────────────────────────────
			//  Anomaly operations
			// ──────────────────────────────────────────────
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				displayOptions: { show: { resource: ['anomaly'] } },
				options: [
					{ name: 'Get Feed', value: 'getFeed', description: 'Get anomaly feed' },
					{ name: 'Explain Single', value: 'explain', description: 'GPT-4 powered anomaly explanation' },
					{ name: 'Explain Batch', value: 'explainBatch', description: 'Analyze multiple anomalies for patterns' },
				],
				default: 'getFeed',
			},
			{
				displayName: 'Anomaly Data (JSON)',
				name: 'anomalyData',
				type: 'json',
				displayOptions: { show: { resource: ['anomaly'], operation: ['explain'] } },
				default: '{}',
				description: 'Anomaly object with type, keyword, severity, etc.',
			},
			{
				displayName: 'Anomalies Array (JSON)',
				name: 'anomaliesArray',
				type: 'json',
				displayOptions: { show: { resource: ['anomaly'], operation: ['explainBatch'] } },
				default: '[]',
				description: 'Array of anomaly objects for batch analysis',
			},

			// ──────────────────────────────────────────────
			//  Optimization operations
			// ──────────────────────────────────────────────
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				displayOptions: { show: { resource: ['optimization'] } },
				options: [
					{ name: 'Generate Plan', value: 'generatePlan', description: 'Generate optimization plan for a campaign' },
				],
				default: 'generatePlan',
			},

			// ──────────────────────────────────────────────
			//  Contextual Bidding operations
			// ──────────────────────────────────────────────
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				displayOptions: { show: { resource: ['contextualBidding'] } },
				options: [
					{ name: 'Optimize Profile', value: 'optimize', description: 'Run contextual Thompson Sampling' },
					{ name: 'Learn from Outcomes', value: 'learn', description: 'Update arm posteriors' },
					{ name: 'Get Arm Stats', value: 'getArmStats', description: 'Per-arm posterior statistics' },
					{ name: 'Get Features', value: 'getFeatures', description: 'Context feature vector for a keyword' },
					{ name: 'Get Feature Importance', value: 'getFeatureImportance', description: 'Weight vectors per arm' },
					{ name: 'Select Arm', value: 'selectArm', description: 'Select best bid multiplier' },
					{ name: 'Get Status', value: 'getStatus', description: 'Optimizer status' },
				],
				default: 'optimize',
			},
			{
				displayName: 'Profile ID',
				name: 'profileId',
				type: 'string',
				displayOptions: {
					show: {
						resource: ['contextualBidding', 'rlBudget'],
						operation: ['optimize', 'learn', 'allocate', 'learnRl', 'getState'],
					},
				},
				default: '',
				required: true,
			},
			{
				displayName: 'Keyword ID',
				name: 'keywordId',
				type: 'number',
				displayOptions: {
					show: {
						resource: ['contextualBidding'],
						operation: ['getArmStats', 'getFeatures', 'getFeatureImportance', 'selectArm'],
					},
				},
				default: 0,
				required: true,
			},
			{
				displayName: 'Dry Run',
				name: 'dryRun',
				type: 'boolean',
				displayOptions: {
					show: {
						resource: ['contextualBidding', 'rlBudget'],
						operation: ['optimize', 'allocate'],
					},
				},
				default: true,
				description: 'Simulate without applying changes',
			},
			{
				displayName: 'Lookback Days',
				name: 'lookbackDays',
				type: 'number',
				displayOptions: {
					show: {
						resource: ['contextualBidding', 'rlBudget'],
						operation: ['learn', 'learnRl'],
					},
				},
				default: 7,
				description: 'Days of historical data for learning',
			},

			// ──────────────────────────────────────────────
			//  RL Budget operations
			// ──────────────────────────────────────────────
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				displayOptions: { show: { resource: ['rlBudget'] } },
				options: [
					{ name: 'Allocate Budget', value: 'allocate', description: 'Run hierarchical RL budget allocation' },
					{ name: 'Trigger Learning', value: 'learnRl', description: 'Policy gradient update' },
					{ name: 'Get State', value: 'getState', description: 'Latest portfolio state' },
				],
				default: 'allocate',
			},
			{
				displayName: 'Total Budget',
				name: 'totalBudget',
				type: 'number',
				displayOptions: { show: { resource: ['rlBudget'], operation: ['allocate'] } },
				default: 500,
				description: 'Total daily budget to distribute ($)',
			},

			// ──────────────────────────────────────────────
			//  Semantic Intelligence operations
			// ──────────────────────────────────────────────
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				displayOptions: { show: { resource: ['semantic'] } },
				options: [
					{ name: 'Detect Bleed', value: 'detectBleed', description: 'Find wasted spend from semantically irrelevant terms' },
					{ name: 'Find Opportunities', value: 'findOpportunities', description: 'Discover high-value untapped terms' },
					{ name: 'Health Report', value: 'healthReport', description: 'Semantic health status for an account' },
				],
				default: 'detectBleed',
			},
			{
				displayName: 'Account ID',
				name: 'accountId',
				type: 'number',
				displayOptions: { show: { resource: ['semantic'] } },
				default: 0,
				required: true,
			},
			{
				displayName: 'ASIN',
				name: 'asin',
				type: 'string',
				displayOptions: {
					show: {
						resource: ['semantic'],
						operation: ['detectBleed', 'findOpportunities'],
					},
				},
				default: '',
				required: true,
			},

			// ──────────────────────────────────────────────
			//  Competitive Intel operations
			// ──────────────────────────────────────────────
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				displayOptions: { show: { resource: ['competitive'] } },
				options: [
					{ name: 'Get Competitors', value: 'getCompetitors', description: 'Fetch competitor intelligence data' },
				],
				default: 'getCompetitors',
			},

			// ──────────────────────────────────────────────
			//  Ingestion operations
			// ──────────────────────────────────────────────
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				displayOptions: { show: { resource: ['ingestion'] } },
				options: [
					{ name: 'Trigger Sync', value: 'triggerSync', description: 'Trigger data ingestion from Amazon APIs' },
				],
				default: 'triggerSync',
			},

			// ──────────────────────────────────────────────
			//  Health operations
			// ──────────────────────────────────────────────
			{
				displayName: 'Operation',
				name: 'operation',
				type: 'options',
				displayOptions: { show: { resource: ['health'] } },
				options: [
					{ name: 'Check', value: 'check', description: 'API health check' },
				],
				default: 'check',
			},
		],
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnData: INodeExecutionData[] = [];
		const credentials = await this.getCredentials('optimusPrimeApi');
		const baseUrl = (credentials.baseUrl as string).replace(/\/$/, '');

		for (let i = 0; i < items.length; i++) {
			try {
				const resource = this.getNodeParameter('resource', i) as string;
				const operation = this.getNodeParameter('operation', i) as string;

				let method = 'GET';
				let endpoint = '';
				let body: Record<string, unknown> = {};
				let qs: Record<string, string> = {};

				// ── Dashboard ────────────────────────────
				if (resource === 'dashboard') {
					if (operation === 'getSummary') {
						endpoint = '/api/v1/dashboard/summary';
					} else if (operation === 'getChartData') {
						endpoint = '/api/v1/dashboard/chart-data';
						qs.range = this.getNodeParameter('timeRange', i) as string;
					} else if (operation === 'getAiActions') {
						endpoint = '/api/v1/dashboard/ai-actions';
					} else if (operation === 'getClientMatrix') {
						endpoint = '/api/v1/dashboard/matrix';
					} else if (operation === 'getClientDashboard') {
						const clientId = this.getNodeParameter('clientId', i) as string;
						endpoint = `/api/v1/performance/dashboard/${clientId}`;
					}
				}

				// ── Campaign ─────────────────────────────
				else if (resource === 'campaign') {
					if (operation === 'list') {
						endpoint = '/api/v1/campaigns';
					} else if (operation === 'updateStrategy') {
						method = 'PATCH';
						const campaignId = this.getNodeParameter('campaignId', i) as string;
						endpoint = `/api/v1/campaigns/${campaignId}/strategy`;
						body = { ai_mode: this.getNodeParameter('aiMode', i) as string };
					}
				}

				// ── Anomaly ──────────────────────────────
				else if (resource === 'anomaly') {
					if (operation === 'getFeed') {
						endpoint = '/api/v1/anomalies/feed';
					} else if (operation === 'explain') {
						method = 'POST';
						endpoint = '/api/v1/anomalies/explain';
						const anomalyData = this.getNodeParameter('anomalyData', i) as string;
						body = { anomaly: typeof anomalyData === 'string' ? JSON.parse(anomalyData) : anomalyData };
					} else if (operation === 'explainBatch') {
						method = 'POST';
						endpoint = '/api/v1/anomalies/explain-batch';
						const arr = this.getNodeParameter('anomaliesArray', i) as string;
						body = { anomalies: typeof arr === 'string' ? JSON.parse(arr) : arr };
					}
				}

				// ── Contextual Bidding ───────────────────
				else if (resource === 'contextualBidding') {
					if (operation === 'optimize') {
						method = 'POST';
						const profileId = this.getNodeParameter('profileId', i) as string;
						endpoint = `/api/v1/contextual-bidding/${profileId}/optimize`;
						body = {
							dry_run: this.getNodeParameter('dryRun', i) as boolean,
							mode: 'contextual',
						};
					} else if (operation === 'learn') {
						method = 'POST';
						const profileId = this.getNodeParameter('profileId', i) as string;
						endpoint = `/api/v1/contextual-bidding/${profileId}/learn`;
						body = {
							lookback_days: this.getNodeParameter('lookbackDays', i) as number,
							mode: 'contextual',
						};
					} else if (operation === 'getArmStats') {
						const keywordId = this.getNodeParameter('keywordId', i) as number;
						endpoint = `/api/v1/contextual-bidding/keywords/${keywordId}/arms`;
					} else if (operation === 'getFeatures') {
						const keywordId = this.getNodeParameter('keywordId', i) as number;
						endpoint = `/api/v1/contextual-bidding/keywords/${keywordId}/features`;
					} else if (operation === 'getFeatureImportance') {
						const keywordId = this.getNodeParameter('keywordId', i) as number;
						endpoint = `/api/v1/contextual-bidding/keywords/${keywordId}/feature-importance`;
					} else if (operation === 'selectArm') {
						method = 'POST';
						const keywordId = this.getNodeParameter('keywordId', i) as number;
						endpoint = `/api/v1/contextual-bidding/keywords/${keywordId}/select-arm`;
						body = {};
					} else if (operation === 'getStatus') {
						endpoint = '/api/v1/contextual-bidding/status';
					}
				}

				// ── RL Budget ────────────────────────────
				else if (resource === 'rlBudget') {
					if (operation === 'allocate') {
						method = 'POST';
						endpoint = '/api/v1/rl-budget/allocate';
						body = {
							profile_id: this.getNodeParameter('profileId', i) as string,
							total_budget: this.getNodeParameter('totalBudget', i) as number,
							dry_run: this.getNodeParameter('dryRun', i) as boolean,
						};
					} else if (operation === 'learnRl') {
						method = 'POST';
						endpoint = '/api/v1/rl-budget/learn';
						body = {
							profile_id: this.getNodeParameter('profileId', i) as string,
							lookback_days: this.getNodeParameter('lookbackDays', i) as number,
						};
					} else if (operation === 'getState') {
						const profileId = this.getNodeParameter('profileId', i) as string;
						endpoint = `/api/v1/rl-budget/state/${profileId}`;
					}
				}

				// ── Semantic Intelligence ────────────────
				else if (resource === 'semantic') {
					const accountId = this.getNodeParameter('accountId', i) as number;
					if (operation === 'detectBleed') {
						const asin = this.getNodeParameter('asin', i) as string;
						endpoint = `/api/v1/semantic/bleed?account_id=${accountId}&asin=${asin}`;
					} else if (operation === 'findOpportunities') {
						const asin = this.getNodeParameter('asin', i) as string;
						endpoint = `/api/v1/semantic/opportunities?account_id=${accountId}&asin=${asin}`;
					} else if (operation === 'healthReport') {
						endpoint = `/api/v1/semantic/health?account_id=${accountId}`;
					}
				}

				// ── Competitive Intel ────────────────────
				else if (resource === 'competitive') {
					endpoint = '/api/v1/competitive/competitors';
				}

				// ── Ingestion ────────────────────────────
				else if (resource === 'ingestion') {
					method = 'POST';
					endpoint = '/api/v1/ingestion/sync';
				}

				// ── Health ───────────────────────────────
				else if (resource === 'health') {
					endpoint = '/health';
				}

				if (!endpoint) {
					throw new NodeOperationError(
						this.getNode(),
						`Unknown resource/operation: ${resource}/${operation}`,
					);
				}

				const url = `${baseUrl}${endpoint}`;
				const options: Record<string, unknown> = {
					method,
					url,
					json: true,
					headers: {
						'Content-Type': 'application/json',
						Authorization: `Bearer ${credentials.apiToken}`,
					},
				};

				if (method !== 'GET' && Object.keys(body).length > 0) {
					options.body = body;
				}
				if (Object.keys(qs).length > 0) {
					options.qs = qs;
				}

				const response = await this.helpers.httpRequest(options as any);
				returnData.push({ json: response });
			} catch (error) {
				if (this.continueOnFail()) {
					returnData.push({
						json: { error: (error as Error).message },
					});
					continue;
				}
				throw error;
			}
		}

		return [returnData];
	}
}
