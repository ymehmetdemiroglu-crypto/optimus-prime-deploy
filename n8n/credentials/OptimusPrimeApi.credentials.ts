import {
	IAuthenticateGeneric,
	ICredentialType,
	INodeProperties,
} from 'n8n-workflow';

export class OptimusPrimeApi implements ICredentialType {
	name = 'optimusPrimeApi';
	displayName = 'Optimus Prime API';
	documentationUrl = 'https://github.com/ymehmetdemiroglu-crypto/optimus-prime-deploy';
	properties: INodeProperties[] = [
		{
			displayName: 'Base URL',
			name: 'baseUrl',
			type: 'string',
			default: 'http://localhost:8000',
			placeholder: 'http://optimus-api:8000',
			description: 'The base URL of the Optimus Prime API server',
		},
		{
			displayName: 'API Token (JWT)',
			name: 'apiToken',
			type: 'string',
			typeOptions: { password: true },
			default: '',
			description: 'JWT bearer token obtained from POST /api/v1/auth/login',
		},
	];

	authenticate: IAuthenticateGeneric = {
		type: 'generic',
		properties: {
			headers: {
				Authorization: '=Bearer {{$credentials.apiToken}}',
			},
		},
	};
}
