#####
#   about: Script to run locally on chroma server to create tenant and db
#  author: Peter Karacsonyi <peterkaracsonyi85@gmail.com>
#    date: 22 Nov 2024
# license: GNU General Public License, version 2
#####



__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from chromadb import AdminClient


class ChromaAdmin:

    def __init__(self, settings=None):
        """
        Initialize the ChromaAdmin with optional settings.
        If no settings are provided, default settings will be used.
        """

        self.client = AdminClient()

    def create_tenant(self, tenant_name: str) -> None:
        """
        Create a new tenant.

        Args:
            tenant_name (str): The name of the tenant to create.
        """
        try:
            self.client.create_tenant(tenant_name)
            print(f"Tenant '{tenant_name}' created successfully.")
        except ValueError as e:
            print(f"Error creating tenant '{tenant_name}': {e}")

    def get_tenant(self, tenant_name: str):
        """
        Retrieve an existing tenant.

        Args:
            tenant_name (str): The name of the tenant to retrieve.

        Returns:
            Tenant object if found, otherwise None.
        """
        try:
            tenant = self.client.get_tenant(tenant_name)
            print(f"Tenant '{tenant_name}' retrieved successfully.")
            return tenant
        except ValueError as e:
            print(f"Error retrieving tenant '{tenant_name}': {e}")
            return None

    def create_database(self, database_name: str, tenant_name: str = "default") -> None:
        """
        Create a new database under a specified tenant.

        Args:
            database_name (str): The name of the database to create.
            tenant_name (str): The tenant under which to create the database.
        """
        try:
            self.client.create_database(database_name, tenant=tenant_name)
            print(f"Database '{database_name}' created under tenant '{tenant_name}' successfully.")
        except ValueError as e:
            print(f"Error creating database '{database_name}' under tenant '{tenant_name}': {e}")

    def get_database(self, database_name: str, tenant_name: str = "default"):
        """
        Retrieve an existing database under a specified tenant.

        Args:
            database_name (str): The name of the database to retrieve.
            tenant_name (str): The tenant under which the database exists.

        Returns:
            Database object if found, otherwise None.
        """
        try:
            database = self.client.get_database(database_name, tenant=tenant_name)
            print(f"Database '{database_name}' retrieved successfully under tenant '{tenant_name}'.")
            return database
        except ValueError as e:
            print(f"Error retrieving database '{database_name}' under tenant '{tenant_name}': {e}")
            return None


if __name__ == "__main__":
    # Initialize the admin client
    admin = ChromaAdmin()

    # Create a tenant
    admin.get_tenant("default")

    # Create a database under a specific tenant
    admin.create_database("azuredocs", tenant_name="default")