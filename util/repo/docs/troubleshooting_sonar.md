## Problems with Sonar?

SonarQube is a sofisticated and intricated tool.  It is not uncommon
that you may have some troubles making it working.  

We have test it in normal conditions and it works properly.  Still, if
you have any of the following issues when running `make sonar`, we
show you a recommendation to solve them.

### `ERROR: Not authorized.`

- **Origin**: It normally arises when running:

		make sonar
	
- **Error message**: You obtain this error:

		ERROR: Not authorized. Please check the properties sonar.login and sonar.password.
		ERROR: 
		ERROR: Re-run SonarQube Scanner using the -X switch to enable full debug logging.

- **Solution**: Edit the `.sonarc` file including the proper credentials, eg:

		sonar.projectKey=iHacker_NewRepo
		sonar.organization=iHacker-github
		sonar.login=2084a54ce06b4d193900141cf67a163681f746d1

- **Other instructions**: See [Repository Personalization](../../../README.md#personalization) in the README.

